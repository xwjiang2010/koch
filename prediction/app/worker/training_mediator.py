import os
from datetime import datetime
from functools import partial
from itertools import count
from typing import Dict, List, Union

import configargparse
import pandas as pd
import ray
from loguru import logger

import prediction.app.worker.ray_processors as ray_proc
import prediction.src.db.prediction_database as pred_db
from prediction.app.prediction_commands import (
    DataCommand,
    HyperparameterTuning,
    TrainingModel,
    TrainModelTrial,
)
from prediction.app.worker.initializer import initialize
from prediction.app.worker.ray_cluster_utils import ray_cluster
from prediction.app.worker.resolvers import (
    resolve_model_command_modeler,
    resolve_model_data_preprocessor,
)
from prediction.app.worker.training_handlers import DeepARModeler, TSAModeler
from prediction.app.worker.utils import cap
from prediction.log_config import setup_logging
from prediction.src import s3_files
from prediction.src.db.prediction_database import (
    TrainRunConfig,
    TrainRunTrial,
    get_train_run_config,
)

# from ray.util.multiprocessing import Pool
CLUSTER_ENV = os.environ.get("CLUSTER_ENV", "prediction-service:19")
LOCAL_CLUSTER = bool(int(os.environ.get("LOCAL_CLUSTER", "0")))


class TuningFailed(Exception):
    pass


class ModelTrainingMediator:
    def __init__(self, config: TrainRunConfig, modelConfigs: List, run_id: int):
        """
        Initialize the database config
        Popualte queues:
        1. Data queue
        2. Model_Queue
        """
        self.TUNE_QUEUE: List = []
        self.DATA_QUEUE_ID: int = None  # Mapping of targets to their ray object ids
        self.MODEL_QUEUE: List = []
        self.trainConfig = config
        self.target_columns = None
        self.run_id = run_id
        logger.info(f"*******TrainRunConfig: {config.train_data_end_dtm}")
        self.dataCommand = DataCommand(
            target_data_location=config.target_data_location,
            feature_data_location=config.feature_data_location,
            target_feature_mapping_location=config.target_feature_mapping_location,
            train_data_end_dtm=config.train_data_end_dtm,
            test_data_end_dtm=config.test_data_end_dtm,
            model_name="ARIMA",
            validation_data_end_dtm=config.validation_data_end_dtm,
            file_type=config.train_data_file_type,
            prediction_steps=config.prediction_steps,
        )
        self.modelConfigList = modelConfigs
        # TODO: Create TrainingModel for each model and place in the MODEL_QUEUE
        for modelConfig in self.modelConfigList:

            self.dataCommand.model_frequency = modelConfig.model_time_interval
            tuneAlg = modelConfig.tune_algorithm
            metrics = (
                self._format_metrics_to_dict(modelConfig.metrics)
                if modelConfig.metrics is not None
                else {}
            )

            if tuneAlg is None or tuneAlg == "":
                modelParams = modelConfig.model_config
                params = modelParams
                hyperparams = {}

                if "hyperparameters" in modelParams:
                    params = modelParams["parameters"]
                    hyperparams = modelParams["hyperparameters"]
                modelCommand = TrainingModel(
                    train_model_id=modelConfig.train_run_model_id,
                    train_name=self.trainConfig.train_name,
                    train_description=self.trainConfig.train_description,
                    train_job_type=self.trainConfig.train_job_type,
                    metrics=metrics,
                    model_location=modelConfig.model_artifacts_location,
                    model_name=modelConfig.model_name,
                    model_config=modelParams,
                    model_metric=modelConfig.model_loss_function,
                    model_time_frequency=modelConfig.model_time_interval,
                    model_params=params,
                    model_hyperparams=hyperparams,
                    modeler=None,
                    data_preprocessor=None,
                    job_name=None,
                    train_run_id=modelConfig.train_run_id.train_run_id,
                )
                resolve_model_command_modeler(modelCommand)
                resolve_model_data_preprocessor(modelCommand)
                modelCommand.db_id = modelConfig.train_run_model_id
                modelCommand.job_name = self._format_job_name(modelCommand)
                logger.info(f"Appending ModelCommand: {modelCommand.model_name}")
                self.MODEL_QUEUE.append(modelCommand)
            else:
                modelParams = modelConfig.model_config
                params = modelParams
                hyperparams = {}
                if "hyperparameters" in modelParams:
                    params = modelParams["parameters"]
                    hyperparams = modelParams["hyperparameters"]

                modelCommand = HyperparameterTuning(
                    train_model_id=modelConfig.train_run_model_id,
                    train_name=self.trainConfig.train_name,
                    train_description=self.trainConfig.train_description,
                    train_job_type=self.trainConfig.train_job_type,
                    metrics=metrics,
                    model_location=modelConfig.model_artifacts_location,
                    model_name=modelConfig.model_name,
                    model_config=modelParams,
                    model_metric=modelConfig.model_loss_function,
                    model_time_frequency=modelConfig.model_time_interval,
                    model_params=params,
                    model_hyperparams=hyperparams,
                    hyperparam_algorithm=modelConfig.tune_algorithm,
                    modeler=None,
                    data_preprocessor=None,
                    job_name=None,
                )
                resolve_model_command_modeler(modelCommand)
                resolve_model_data_preprocessor(modelCommand)
                modelCommand.db_id = modelConfig.train_run_model_id
                modelCommand.job_name = self._format_job_name(modelCommand)
                # logger.info(f"Appending ModelCommand: {modelCommand}")
                self.MODEL_QUEUE.append(modelCommand)

    # ----------------Train------------------------------------------
    def train_in_parallel(self):
        # STEP 1: Populate data queue with the data
        startTs = datetime.now()
        pred_db.update_train_status(
            self.trainConfig.train_run_id, status="Running", startTs=startTs
        )
        status = "Failed"
        try:
            with ray_cluster("train", os.environ.get("CLUSTER_ID", self.run_id)):
                # self._prepare_model_data()
                tasks = []
                data_command = self.dataCommand
                for model_command in self.MODEL_QUEUE:

                    self._prepare_model_data(model_command)

                    pred_db.update_train_model_config_job_name(
                        model_command.train_model_id, model_command.job_name
                    )
                    if model_command.modeler:
                        modeler = model_command.modeler(
                            target_references_id=self.DATA_QUEUE_ID
                        )
                    else:
                        raise NotImplementedError(
                            "New model type not yet added to conditional"
                        )
                    for target in self.target_columns:
                        trainer = ParallelTrain.remote(
                            modeler, model_command, data_command, target
                        )
                        logger.info(f"Training Model: {model_command.model_name}")
                        tasks.append(trainer.train_single_target.remote())

                logger.info("Waiting for tasks")
                total = len(tasks)
                completed = count(1)
                while tasks:
                    ready, tasks = ray.wait(tasks)
                    logger.debug(f"Completed {next(completed)}/{total}")
                    result = ray.get(ready)[0]
                    logger.info(f"{result=}")
                    pred_db.create_train_run_trial(**result)

                status = "Trained"
        finally:
            pred_db.update_train_status(
                self.trainConfig.train_run_id, status=status, startTs=startTs
            )

    def _prepare_model_data(self, modelCommand: TrainingModel, isSplit: bool = True):
        """Pushes data to the shared object store with Ray - Optionally splits the data"""
        data_preprocessor = modelCommand.data_preprocessor()
        data_dict = data_preprocessor.handle(self.dataCommand, modelCommand)
        data_references = {}
        logger.info(
            f"******TrainRunID: {self.trainConfig.train_run_id}"
            "-- DATA PREP: START*************"
        )
        self.target_columns = list(data_dict)
        if modelCommand.model_name == "DEEPAR":
            data_references["DEEPAR"] = {
                "data": ray.put(data_dict),
                "features": None,
            }
        else:
            for target_column, target_data_dict in data_dict.items():
                df = target_data_dict["data"]
                df_features = target_data_dict["features"]
                split_data = {}
                if isSplit:
                    (
                        train_df,
                        test_df,
                        validation_df,
                    ) = data_preprocessor.split_train_test_valid(
                        df=df, request=self.dataCommand
                    )
                    split_data["train"] = ray.put(train_df)
                    split_data["test"] = ray.put(test_df)
                    split_data["valid"] = ray.put(validation_df)
                else:
                    split_data["train"] = ray.put(df)
                data_references[target_column] = {
                    "data": split_data,
                    "features": ray.put(df_features),
                }
                # logger.info(f"******_prepare_model_data::dataTuple: {dataTuple}")
                logger.debug(
                    f"******TrainRunID: {self.trainConfig.train_run_id}"
                    f"-- DATA PREP for {target_column} DONE*************"
                )
        self.DATA_QUEUE_ID = ray.put(data_references)

        logger.info(
            f"******TrainRunID: {self.trainConfig.train_run_id}"
            "-- DATA PREP: END*************"
        )
        # data = DATA_QUEUE[0]["data"]
        # data.Date = pd.to_datetime(data.Date, errors = 'coerce')
        # logging.warning(f'Training data: {data}')

    # ------------------Ray Tune: TUNING------------------------------

    def tune_model(
        self, model_command, modeler: Union[TSAModeler, DeepARModeler]
    ) -> Dict:
        # modelErrors = self._train_test_validate(dfSet, modelCommand)
        rayConfig = ray_proc.RayTuneConfigCreator()
        model_start_ts = datetime.now()
        modelName = model_command.model_name
        run_id = self.trainConfig.train_run_id
        ref = f"{run_id}-{modelName}"
        logger.info(
            f"**************Tuning-{ref}: step 2 - Creating tune config for "
            f"{modelName}: {model_command}"
        )
        try:
            tuneConfig = rayConfig.handle(model_command)
            logger.info(
                f"**********Tuning-{ref}: step 2 - "
                f"Created tune model config: {tuneConfig}"
            )
            model_command.ray_tune_config = tuneConfig
            tuneExec = ray_proc.RayTuneExecutor.remote()
            # TODO :: Group by target column and create records for each target column
            # This may be helpful df = pd.concat(analysis.trial_dataframes.values())
            tune_exec_obj_ref = tuneExec.handle.remote(
                request=model_command,
                modeler=modeler,
                tuneConfig=tuneConfig,
                targets=self.target_columns,
            )
            experiment_dfs = ray.get(tune_exec_obj_ref)
            trials = []
            for target, experiment_df in experiment_dfs.items():
                metrics = get_metrics(experiment_df)

                logger.debug(
                    f"*****Tuning-{ref} Best: acc = {metrics['best_acc']}, config = {metrics['best_config']}"
                )
                trial = TrainModelTrial(
                    target_column=target,
                    model_loss_function=model_command.model_metric,
                    model_config=metrics["best_config"],
                    trial_status="Completed",
                    train_score=metrics["best_acc"],
                    test_score=metrics["best_acc"],
                    validation_score=metrics["best_validation_acc"],
                    trial_start_ts=model_start_ts,
                    trial_end_ts=datetime.now(),
                    metrics=metrics["extra"],
                )
                trials.append(
                    {"trial": trial, "train_run_model_id": model_command.db_id}
                )
            return trials
        except Exception as e:
            logger.error(f"Error in executing Ray Tune for {modelName}")
            logger.exception(e)

    def tune(self):
        with ray_cluster(
            action="tune", run_id=os.environ.get("CLUSTER_ID", self.run_id)
        ):
            # TODO :: abstract the job queueing so we can re-use it
            # in training and predictions.
            # STEP 1: Populate data queue with the data
            logger.info("**********Tuning: step 1 - Preparing data for model")
            startTs = datetime.now()
            run_id = self.trainConfig.train_run_id
            pred_db.update_train_status(
                self.trainConfig.train_run_id, status="Running", startTs=startTs
            )

            # Submit tasks
            logger.info(f"Tuning-{run_id}")
            for model_command in self.MODEL_QUEUE:
                with logger.contextualize(model_name=model_command.model_name):
                    logger.info(f"Begin tuning of model: {model_command.model_name}")
                    self._prepare_model_data(model_command, isSplit=True)

                    if model_command.modeler:
                        modeler = model_command.modeler(
                            target_references_id=self.DATA_QUEUE_ID
                        )
                    else:
                        raise NotImplementedError(
                            "New model type not yet added to conditional"
                        )

                    # TODO :: allow passing in the task so we can
                    # leave this open of heterogenous
                    # tasks running on the same set of workers / clusters.
                    results = self.tune_model(
                        model_command=model_command, modeler=modeler
                    )
                    logger.info(
                        f"Completed tuning of model: {model_command.model_name}"
                    )
                    if not results:
                        logger.error("Tuning Failed")
                        raise TuningFailed
                    for result in results:
                        if result is not None:
                            pred_db.create_train_run_trial(**result)

            logger.info(f"Tuning-{run_id} completed")

    def _format_metrics_to_dict(self, metrics: List[Dict]) -> Dict:
        return {metric["name"]: metric.get("config", True) for metric in metrics}

    def _format_job_name(
        self, model_command: Union[TrainingModel, HyperparameterTuning]
    ):
        return f'{model_command.model_name}-{datetime.now().strftime("%Y%m%d%H%M")}'


@ray.remote(num_cpus=1)
class ParallelTrain:
    def __init__(
        self,
        modeler: TSAModeler,
        model_command: TrainingModel,
        data_command: DataCommand,
        target: str,
    ):
        self.handler = modeler
        self.model_command = model_command
        self.data_command = data_command
        self.target = target
        self.status = None
        setup_logging()

    def train_test_validate(self):
        logger.info(self.model_command)
        if self.model_command.model_name == "DEEPAR":
            self.handler.load_data(self.model_command, self.data_command)

        error, validation_error, metrics = self.handler.handle(
            request=self.model_command, target=self.target
        )
        logger.info(f"Error: {error}, {validation_error=}")

        # TODO FIX :: return the distinct train and test scores
        return {
            "train_score": error,
            "test_score": error,
            "validation_score": validation_error,
            "metrics": metrics,
        }

    def train_single_target(self) -> Dict:
        # STEP 2: Run training job using DATA_QUEUE and MODEL_QUEUE
        status = "Completed"
        modelStartTs = datetime.now()
        with logger.contextualize(
            model=self.model_command.model_name, target=self.target
        ):
            logger.info(
                f"____Ray Training {self.model_command.model_name}-{self.target}: {self.model_command}"
            )
            # TODO Parallelize
            train_test_validate = self.train_test_validate()

            endTs = datetime.now()
            # Update the database----------------
            # Store trial info
            trial = TrainModelTrial(
                target_column=self.target,
                model_loss_function=self.model_command.model_metric,
                model_config=self.model_command.model_config,
                trial_status=status,
                train_score=train_test_validate["train_score"],
                test_score=train_test_validate["test_score"],
                validation_score=train_test_validate["validation_score"],
                trial_start_ts=modelStartTs,
                trial_end_ts=endTs,
                metrics=train_test_validate["metrics"],
            )
            return {"trial": trial, "train_run_model_id": self.model_command.db_id}

    def get_status(self):
        return self.status


def get_metrics(experiment_df: pd.DataFrame) -> Dict:
    # Find the best trial by mean_accuracy
    best_trial = experiment_df[
        experiment_df.mean_accuracy == experiment_df.mean_accuracy.max()
    ].to_dict(orient="records")[0]

    # extracts the config from the flattened structure found in the DataFrame
    config = {
        col.replace("config/", ""): val
        for col, val in best_trial.items()
        if "config/" in col
    }
    metrics = {
        "best_config": config,
        "best_validation_acc": best_trial["validation_accuracy"],
        "best_acc": best_trial["mean_accuracy"],
    }

    metrics["extra"] = {}

    if best_trial.get("bias", False):
        metrics["extra"]["bias"] = best_trial.get("bias")

    return metrics


def write_train_results(
    file_path: str, training_models_dict: Dict, trials: List[TrainRunTrial]
):
    #  Generate data for saving to S3----------------

    # TODO :: This is really weird we should create a db
    # query instead of joining with dictionary operations
    # For now I'm trying to reduce the amount of code I need to modify - Devin
    trainStatus = {}
    modelNum = 1
    for trial in trials:
        logger.info(
            "TrainTrial: Getting TrainModelConfig for"
            f" {trial.train_run_model_id.train_run_model_id}"
        )
        model = training_models_dict[trial.train_run_model_id.train_run_model_id]
        modelName = model.model_name
        loss_function = model.model_loss_function or "MAPE"
        capped = partial(cap, loss_function)

        key = "-".join([modelName, str(modelNum)])
        trainStatus[key] = {
            "target_column": trial.target_column_name,
            "train_start_ts": trial.trial_start_ts,
            "train_end_ts": trial.trial_end_ts,
            "model": modelName,
            "model_params": trial.train_model_config,
            "train_score": capped(trial.train_score),
            "test_score": capped(trial.test_score),
            "validation_score": capped(trial.validation_score),
            "metric": model.metrics,
        }
        modelNum += 1

    df = pd.DataFrame.from_dict(trainStatus, orient="index")
    s3_files.save_train_result(df=df, s3Path=file_path)


def _dispatch_tasks(args: configargparse.Namespace):
    run_id = args.run_id
    action = args.action
    with logger.contextualize(run_id=run_id, action=action):
        logger.info(
            f"------------------------------- STARTING {action}------------------"
        )
        try:
            logger.info(f"*********Parallel {action} for runID = {run_id}")
            train_run_config = get_train_run_config(run_id)
            logger.debug(f"*********Model {action} run config: {train_run_config}")
            model_configs = pred_db.get_model_configs_by_train_id(run_id)
            logger.debug(f"**********Model configs: {model_configs}")
            logger.info(f"**********Total # of models: {len(model_configs)}")
            mediator = ModelTrainingMediator(train_run_config, model_configs, run_id)
            status = "Failed"

            if action == "train":  # TODO:
                logger.info("*********Calling TRAINING in Parallel")
                mediator.train_in_parallel()
            elif action == "tune":
                logger.info("*********Calling TUNING in Parallel")
                mediator.tune()
            # pred_db.update_train_status(runID, status="Trained")
            trials = pred_db.get_train_run_trials_by_train_id(run_id)
            train_models_dict = {
                model_config.train_run_model_id: model_config
                for model_config in model_configs
            }

            #  Create file name
            prefix = f"{train_run_config.train_name}_{train_run_config.train_run_uuid}"
            file_name = s3_files.create_s3_path(
                s3Folder=train_run_config.score_data_location,
                trainName=prefix,
                trainTs=train_run_config.train_start_ts,
            )

            write_train_results(file_name, train_models_dict, trials)
            status = "Completed"
            pred_db.update_train_status(
                trainRunId=run_id,
                status=status,
                endTs=datetime.now(),
                resultPath=file_name,
            )

            logger.info(
                f"------------------------------- COMPLETED {action}------------------"
            )
        except Exception as e:
            pred_db.update_train_status(run_id, status="Failed")
            logger.error(
                f"------------------------------- Failed {action}------------------"
            )
            logger.error(
                f"*********** Exception in {action} using Ray for runId: {run_id}"
            )
            logger.exception(e)


if __name__ == "__main__":
    args = initialize(actions=["train", "tune"])

    with logger.contextualize(user=args.user, run_id=args.run_id):
        logger.info(
            "-------------------------------STARTING RAY TRAINING------------------"
        )

        logger.info(
            "-------------------------------STARTED RAY TRAINING------------------"
        )

        _dispatch_tasks(args)
