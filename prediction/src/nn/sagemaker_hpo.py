import json

import boto3
import sagemaker
from loguru import logger
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.predictor import Predictor
from sagemaker.pytorch import PyTorch
from sagemaker.serializers import IdentitySerializer
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter

# from asyncio.log import logger
# from logging import Logger


class SageMakerTrainer:
    def __init__(self, config, tunable_config, hyperparameters):
        self.config = config
        self.pred_length = self.config["pred_steps"]
        self.frequency = self.config["FREQ"]
        self.train_instance_count = self.config["train_instance_count"]
        self.train_instance_type = self.config["train_instance_type"]
        self.objective_metric_name = self.config["objective_metric_name"]
        self.max_jobs = self.config["max_jobs"]
        self.max_parallel_jobs = self.config["max_parallel_jobs"]
        self.tags = self.config.get("tags", {})
        self.iam_role = config.get("iam_role", None)
        self.hyperparameters = hyperparameters
        self.tunable = tunable_config
        self.tunable_params = {
            "mini_batch_size": IntegerParameter(
                *self.tunable["mini_batch_size"], 20000
            ),  # TODO: Verify and configure max values of tunable param from DS
            "epochs": IntegerParameter(*self.tunable["epochs"], 300),
            "context_length": IntegerParameter(*self.tunable["context_length"], 10),
            "num_cells": IntegerParameter(*self.tunable["num_cells"], 20),
            "num_layers": IntegerParameter(*self.tunable["num_layers"], 5),
            "dropout_rate": ContinuousParameter(*self.tunable["dropout_rate"], 0.10),
            "embedding_dimension": IntegerParameter(
                *self.tunable["embedding_dimension"], 15
            ),
            "learning_rate": ContinuousParameter(
                *self.tunable["learning_rate"], 1.0e-2
            ),
        }

        self.job_name = config.get("job_name", None)
        self.num_samples = config.get("num_samples", None)
        self.tuner = None
        self.predictor = None

    def fit(self, X, y, input_channels, job_name, output_path):
        """
        Training function for LSTM
        """
        if job_name:
            self.job_name = job_name
        sess = sagemaker.Session()
        if not self.iam_role:
            self.iam_role = get_execution_role()

        estimator = Estimator(
            sagemaker_session=sess,
            image_uri=sagemaker.image_uris.retrieve(
                framework="forecasting-deepar",
                version="latest",
                region=sess.boto_region_name,
            ),
            role=self.iam_role,
            train_instance_count=self.train_instance_count,
            train_instance_type=self.train_instance_type,
            output_path=output_path,
            tags=self.tags,
        )
        estimator.set_hyperparameters(**self.hyperparameters)
        status = self._fit_sagemaker(
            estimator=estimator,
            tunable_params=self.tunable_params,
            input_channels=input_channels,
            job_name=self.job_name,
        )
        if status and status.lower() == "completed":
            self.deploy_best_model(self.tuner)

        return status

    def _fit_sagemaker(
        self,
        estimator,
        tunable_params,
        input_channels,
        job_name,
        metric_definitions=None,
    ):

        # Configure HyperparameterTuner
        self.tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name=self.objective_metric_name,
            objective_type="Minimize",
            hyperparameter_ranges=tunable_params,
            metric_definitions=metric_definitions,
            max_jobs=self.max_jobs,
            max_parallel_jobs=self.max_parallel_jobs,
            tags=self.tags,
        )
        # Start hyperparameter tuning job
        print(f"DeepAR training start for job: {job_name}")
        self.tuner.fit(
            inputs=input_channels, include_cls_metadata=False, job_name=job_name
        )
        # Checking tuning job progress
        self.check_for_progress_and_sleep(job_name, 300)
        # Get the final status
        status = SageMakerTrainer.get_job_status(job_name)

        return status

    def get_tuner(self, job_name):
        if job_name:
            self.job_name = job_name
        if not self.tuner and self.job_name:
            self.tuner = HyperparameterTuner.attach(self.job_name)

        return self.tuner

    def deploy_best_model(self, tuner, job_name=None):
        # Deploy best model and return predictor (RealTimePredictor)
        if tuner:
            self.tuner = tuner
        else:
            tuner = self.get_tuner(job_name)

        self.predictor = tuner.deploy(
            initial_instance_count=self.config["pred_instance_count"],
            instance_type=self.config["pred_instance_type"],
        )

    def check_for_progress_and_sleep(self, job_name, sleep_time=120):
        while SageMakerTrainer.check_for_progress(job_name):
            tuning_results = SageMakerTrainer.get_job_statuses(job_name)
            time.sleep(sleep_time)

    def predict_on_request(self, line_record, pred_length=None):

        endpoints = self.get_endpoints(self.job_name)
        if len(endpoints) == 0:
            logger.info("Endpoint doesn't exist, creating Endpoint.")
            self.deploy_best_model(self.tuner, self.job_name)
            logger.info("Successfully deployed model on Endpoint")

        if not self.predictor:
            tuner = self.get_tuner(job_name=None)
            endpoint_name = tuner.best_training_job()
            self.predictor = Predictor(
                endpoint_name, serializer=IdentitySerializer("application/json")
            )

        configuration = {
            "num_samples": self.num_samples,
            "output_types": ["mean", "quantiles", "samples"],
            "quantiles": ["0.1", "0.5", "0.9"],
        }
        instance = {}
        instance["start"] = line_record["start"]
        instance["target"] = line_record["target"][:-pred_length]
        if line_record.get("cat", False):
            instance["cat"] = line_record["cat"]
        if line_record.get("dynamic_feat", False):
            instance["dynamic_feat"] = line_record["dynamic_feat"]

        http_request_data = {"instances": [instance], "configuration": configuration}
        pred = self.predictor.predict(
            json.dumps(http_request_data).encode("utf-8")
        ).decode("utf-8")
        preds = json.loads(pred)

        logger.info("Deleting Endpoint.")
        self.predictor.delete_endpoint(endpoint_name)
        return preds

    @staticmethod
    def get_job_statuses(job_name):
        job_statuses = boto3.client("sagemaker").describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=job_name
        )
        return job_statuses

    @staticmethod
    def get_job_status(job_name, status="HyperParameterTuningJobStatus"):
        try:
            job_statuses = SageMakerTrainer.get_job_statuses(job_name)
            job_status = job_statuses[status]
        except Exception as e:
            logger.error(f"Could not get the status of the job")
            logger.exception(e)
            job_status = "Untraceable"
        return job_status

    @staticmethod
    def check_for_progress(job_name):
        job_status = SageMakerTrainer.get_job_status(job_name)
        if job_status == "InProgress":
            return True
        else:
            return False

    @staticmethod
    def check_for_progress_handle(job_name):
        job_status = SageMakerTrainer.get_job_status(job_name)
        if job_status == "InProgress":
            return "InProgress"
        elif job_status == "Untraceable":
            return "Untraceable"
        else:
            return "NotInProgress"

    def stop_running_process(job_name):
        job_status = SageMakerTrainer.check_for_progress_handle(job_name)
        sagemaker_client = boto3.client("sagemaker")
        if job_status == "InProgress":
            try:
                response = sagemaker_client.stop_hyper_parameter_tuning_job(
                    HyperParameterTuningJobName=job_name
                )
                if response is None:
                    return "SUCCESS"
            except Exception as e:
                return {"error": e}
        elif job_status == "NotInProgress":
            return "STOPPED"
        else:
            return "UNTRACEABLE"

    def remove_endpoint(job_name):
        sagemaker_client = boto3.client("sagemaker")
        try:
            job_details = SageMakerTrainer.get_job_statuses(job_name)
            best_training_job = job_details["BestTrainingJob"]
            endpoint_name = best_training_job["TrainingJobName"]
            response = sagemaker_client.delete_endpoint(endpoint_name)
            if response is None:
                return True
            else:
                return False
        except Exception as e:
            return {"error": e}

    def get_endpoints(job_name):
        endpoints = boto3.client("sagemaker").list_endpoints(NameContains=job_name)
        endpoints = endpoints.get("Endpoints")
        return endpoints
