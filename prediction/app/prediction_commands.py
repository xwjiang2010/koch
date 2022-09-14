import json
import os
import shlex
import subprocess  # nosec
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import backoff
from kubernetes import client, config
from loguru import logger

from prediction.log_config import JSON_LOGS, LOG_LEVEL
from prediction.src.static import PREDICTION_CONTEXT

# -----------------------KUBERNETES CONFIG for RAY-------------------------------------
IMAGE = os.environ.get(
    "CI_REGISTRY_IMAGE",
)
IMAGE_PULL_SECRETS = json.loads(
    os.environ.get(
        "IMAGE_PULL_SECRETS",
    )
    or "{}"
)
NAMESPACE = os.environ.get(
    "KUBE_NAMESPACE",
    "default",
)
DATABASE_URL = os.environ.get("DATABASE_URL", "WRONG DB URL")
TUNE_DISABLE_AUTO_CALLBACK_LOGGERS = os.environ.get(
    "TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", 1
)
CLUSTER_ENV = os.environ.get("CLUSTER_ENV", "prediction-service:19")
ANYSCALE_CLI_TOKEN = os.environ.get("ANYSCALE_CLI_TOKEN")
IAM_INSTANCE_PROFILE_ARN = os.environ.get("IAM_INSTANCE_PROFILE_ARN")

logger.info(f"Ray Image: {IMAGE}")


def get_job_state(status: client.V1JobStatus) -> str:
    if status.start_time is None:
        return "Submitted"
    elif status.active:
        return "Active"
    elif isinstance(status.conditions, list):
        # Either 'Failed' or 'Complete'
        return status.conditions[0].type
    else:
        return "Unknown"


def _job_not_ready(status: client.V1JobStatus) -> bool:
    # We only have a single pod so we can check the first element
    state = get_job_state(status)
    return state not in ["Failed", "Complete", "Submitted", "Active"]


@dataclass
class Command:
    evt_id: uuid.UUID = field(init=False)
    evt_create_ts: datetime = field(init=False)
    evt_processing_start_ts: datetime = field(init=False)
    evt_processing_end_ts: datetime = field(init=False)
    db_id: int = field(init=False)

    def __post_init__(self):
        self.evt_id: uuid.UUID = uuid.uuid4()
        self.evt_create_ts: datetime = datetime.now()
        self.evt_processing_start_ts = None
        self.evt_processing_end_ts = None
        self.db_id = None
        # evt_processing_time: timedelta = None


@dataclass
class DataCommand(Command):
    target_data_location: str
    feature_data_location: str
    model_name: str = "ARIMA"
    train_data_end_dtm: datetime = None
    test_data_end_dtm: datetime = None
    validation_data_end_dtm: datetime = None
    feature_mapping: str = None
    target_feature_mapping_location: Optional[str] = ""
    file_type: str = "csv"
    model_frequency: str = PREDICTION_CONTEXT.default_time_interval
    prediction_steps: str = None


@dataclass
class TrainingModel(Command):
    train_model_id: str
    train_name: str
    train_description: str
    train_job_type: str  # TrainJobType
    model_location: str
    model_name: str
    model_config: Dict
    model_metric: str
    model_time_frequency: str
    model_params: Dict
    model_hyperparams: Dict  # = field(init = False)
    modeler: object
    data_preprocessor: object
    train_run_id: int = None
    metrics: Dict = None
    sample_weights: List[int] = 1
    job_name: str = None


@dataclass
class HyperparameterTuning(TrainingModel):
    hyperparam_algorithm: str = None
    ray_tune_config: Dict = None
    modeler: object

    def __post_init__(self):
        super().__post_init__()
        arimaParams = {"p": [1, 7], "d": [1, 2], "q": [1, 2]}
        holtParams = {
            "trend": ["add", "mul"],
            "damped": [True, False],
            "seasonal_periods": 3,
            "seasonal": [None],
        }
        prophetParams = {
            "growth": ["linear"],
            "changepoints": [None],
            "n_changepoints": [20, 22],
            "changepoint_range": [0.8, 0.9],
            "changepoint_prior_scale": [0.05, 0.1],
            "yearly_seasonality": ["auto"],
            "weekly_seasonality": ["auto"],
            "daily_seasonality": ["auto"],
            "holidays": [None],
            "seasonality_mode": ["additive", "multiplicative"],
            "seasonality_prior_scale": [10.0],
            "holidays_prior_scale": [10.0],
            "interval_width": [0.8],
            "uncertainty_samples": [1000],
        }
        deepar_params = {}  # TBD

        if self.model_params is None or not self.model_params:
            if self.model_name == "ARIMA":
                self.model_params = arimaParams
            if self.model_name == "HOLTWINTERS":
                self.model_params = holtParams
            if self.model_name == "PROPHET":
                self.model_params = prophetParams
            if self.model_name == "DEEPAR":
                self.model_name = deepar_params


@dataclass
class TrainModelTrial(Command):
    target_column: str
    model_loss_function: str
    model_config: Dict
    trial_status: str
    trial_start_ts: datetime
    metrics: Dict = None
    feature_columns: List[str] = None
    train_score: float = None
    test_score: float = None
    validation_score: float = None
    model_trial_location: str = ""
    trial_end_ts: datetime = None


@dataclass
class ScoringCommand(Command):
    score_name: str
    model_name: str
    model_config: Dict
    score_status: str
    prediction_steps: int
    score_time_frequency: str
    target_column: str = ""
    model_version: str = ""
    model_location: str = ""
    model_params: Dict = None
    model_hyperparams: Dict = None
    score_loss_function: str = PREDICTION_CONTEXT.default_loss_function
    prediction_count: int = PREDICTION_CONTEXT.default_prediction_count
    train_score: float = None
    test_score: float = None
    validation_score: float = None
    score_start_ts: datetime = None
    score_end_ts: datetime = None
    score_description: str = ""
    modeler: object = None
    data_preprocessor: object = None
    train_run_id: int = None
    job_name: str = None

    def __post_init__(self):
        super().__post_init__()
        if self.model_config is not None:
            if self.model_params is None:
                self.model_params = self.model_config.get("parameters", None)
            if self.model_hyperparams is None:
                self.model_hyperparams = self.model_config.get("hyperparameters", None)


@dataclass
class RayCommand(Command):
    run_id: int
    run_uuid: uuid.UUID
    action: str = "tune"
    entry_point: str = "prediction.app.worker.training_mediator"
    name: str = field(init=False)
    python_command: str = field(init=False)
    username: str = "nobody"


@dataclass
class RayDistributed(RayCommand):
    """This class brings up the containers and trains in a distributed manner"""

    cpu: str = "6"
    failed_deployment_message: str = "Failed deployment"
    job: client.V1Job = field(init=False)
    memory: str = "1Gi"
    namespace: str = field(default=NAMESPACE)
    number_of_workers: int = 3
    successful_deployment_message: str = "Deployment successful"
    name: str = None

    def __post_init__(self):
        super().__post_init__()
        # Configureate Pod template container
        # TODO :: We should fork our logic here, or rather call an abstracted function
        # that will either launch the job on a cluster or run inside of a process locally.
        config.load_incluster_config()
        self.job_api = client.BatchV1Api()
        self.custom_api = client.CustomObjectsApi()
        if self.name is None:
            self.name = f"ray-client-{self.run_id}-{self.run_uuid}"
        self.python_command = (
            f"python -m {self.entry_point} --run-id={self.run_id}"
            f" --action='{self.action}' --user={self.username}"
        )

        # TODO:: FIX :: Figure out the proper way to shorten these names.
        # Currently this is causing an error since the operator cannot handle names
        # longer than 63 characters, for now we're using the runId instead of the uuid
        command = shlex.split(self.python_command)

        db_env = client.V1EnvVar(name="DATABASE_URL", value=DATABASE_URL)
        log_level_env = client.V1EnvVar(name="LOG_LEVEL", value=LOG_LEVEL)

        image_pull_secrets = client.V1EnvVar(
            name="IMAGE_PULL_SECRETS",
            value=json.dumps(IMAGE_PULL_SECRETS),
        )
        image = client.V1EnvVar(name="CI_REGISTRY_IMAGE", value=IMAGE)
        json_logs_env = client.V1EnvVar(name="JSON_LOGS", value=str(int(JSON_LOGS)))
        cluster_env = client.V1EnvVar(name="CLUSTER_ENV", value=CLUSTER_ENV)
        iam_instance_profile = client.V1EnvVar(
            name="IAM_INSTANCE_PROFILE_ARN", value=IAM_INSTANCE_PROFILE_ARN
        )
        anyscale_token = client.V1EnvVar(
            name="ANYSCALE_CLI_TOKEN", value=ANYSCALE_CLI_TOKEN
        )
        kube_namespace = client.V1EnvVar(name="KUBE_NAMESPACE", value=NAMESPACE)
        resources = client.V1ResourceRequirements(
            requests={"cpu": "1", "memory": "1Gi"}
        )
        container = client.V1Container(
            name="prediction-service-job",
            image=IMAGE,
            command=command,
            env=[
                db_env,
                log_level_env,
                json_logs_env,
                cluster_env,
                anyscale_token,
                iam_instance_profile,
                image,
                image_pull_secrets,
                kube_namespace,
            ],
            resources=resources,
        )
        # Create and configurate a spec section
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": "ray-client"}),
            spec=client.V1PodSpec(
                restart_policy="Never",
                image_pull_secrets=IMAGE_PULL_SECRETS,
                containers=[container],
            ),
        )
        # Create the specification of deployment
        spec = client.V1JobSpec(
            template=template, backoff_limit=4, ttl_seconds_after_finished=1200
        )
        # Instantiate the job object
        self.job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=self.name, namespace=self.namespace),
            spec=spec,
        )

    def _launch_job(self):
        job = self.job_api.create_namespaced_job(
            body=self.job, namespace=self.namespace
        )
        # Update job so we can get the generated name
        self.job = job
        logger.info(f"Job created. status='{job.status}'")

    def _delete_job(self):
        try:
            api_response = self.job_api.delete_namespaced_job(
                name=self.job.metadata.name,
                namespace=self.job.metadata.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=5
                ),
            )
            logger.info(f"Job deleted. status='{api_response.status}'")
        except client.exceptions.ApiException as e:
            logger.error(f"Failed to delete job {self.job.metadata.name}")
            if e.status == 404:
                logger.error(f"Job not found {e}")
            else:
                raise e

    # TODO :: consider adding a timeout
    @backoff.on_predicate(backoff.constant, _job_not_ready, interval=60)
    def _get_job_status(self) -> client.V1JobStatus:
        # We only have one pod, so we can select the first element
        job = self.job_api.read_namespaced_job_status(
            name=self.job.metadata.name, namespace=self.job.metadata.namespace
        )
        status = job.status
        logger.debug(f"Job status: {status}")
        return status

    def run(self) -> str:
        self._launch_job()
        # TODO :: Consider logging more information here
        return get_job_state(self._get_job_status())

    def cleanup(self):
        """This function is used to delete the resources once the jobs are completed"""
        self._delete_job()


@dataclass
class RayLocal(RayCommand):
    def __post_init__(self):
        super().__post_init__()
        # Configureate Pod template container
        # TODO :: We should fork our logic here, or rather call an abstracted function
        # that will either launch the job on a cluster or run inside of a process locally.

        self.name = f"ray-client-{self.run_id}-{self.run_uuid}"
        self.python_command = (
            f"python -m {self.entry_point} --run-id={self.run_id}"
            f" --action='{self.action}' --user={self.username}"
        )

    def run(self) -> str:
        command = shlex.split(self.python_command)
        self._process = subprocess.Popen(command)  # nosec

    def cleanup(self):
        """This function is used to delete the resources once the jobs are completed"""
        if self._process:
            try:
                return_code = self._process.poll()
                if return_code is None:
                    self._process.kill()
                else:
                    logger.info(f"Proccess already exited with {return_code}")
            except Exception as e:
                logger.error("Error occurred while cleaning up Ray Client Job")
                logger.exception(e)


@dataclass
class RayJobHandler:
    name: str = None
    namespace: str = NAMESPACE

    def ray_job_status(self):
        try:
            v1Client = client.BatchV1Api()
            job_status = v1Client.read_namespaced_job_status(
                name=self.name, namespace=self.namespace
            )
            status_as_v1JobClient = job_status.status
            logger.debug(f"Job status: {status_as_v1JobClient}")
            status = get_job_state(status_as_v1JobClient)
        except Exception as e:
            logger.error(f"Could not get the status of the job")
            logger.exception(e)
            status = "Untraceable"
        return status

    def ray_job_delete(self):
        try:
            v1Client = client.BatchV1Api()
            api_response = v1Client.delete_namespaced_job(
                name=self.name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=60
                ),
            )
            delete_status = "Success"
            logger.info(f"Job deleted. status='{api_response.status}'")
        except client.exceptions.ApiException as e:
            delete_status = "Failed"
            logger.error(f"Failed to delete job")
            if e.status == 404:
                logger.error(f"Job not found {e}")
            else:
                raise e
        return delete_status
