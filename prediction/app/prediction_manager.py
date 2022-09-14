import os
from datetime import datetime
from uuid import UUID

from loguru import logger

import prediction.src.db.prediction_database as pred_db
from prediction.app.prediction_commands import (
    HyperparameterTuning,
    RayCommand,
    RayDistributed,
    RayJobHandler,
    RayLocal,
    TrainingModel,
)
from prediction.app.worker.scoring_mediator import ScoringMediator
from prediction.app.worker.training_mediator import ModelTrainingMediator
from prediction.src.static import PREDICTION_CONTEXT

COMMAND_HANDLERS = {
    TrainingModel: ModelTrainingMediator,
    HyperparameterTuning: ModelTrainingMediator,
}
DB_HANDLERS = {TrainingModel: pred_db.create_train_run_config}

RUN_LOCAL = bool(os.environ.get("RUN_LOCAL", 0))


class PredictionMediator:
    def __init__(self, params: dict, isTrain: bool = True, username: str = ""):
        ##self.command = dataclass_from_dict(Train, params)
        ##self.handler = COMMAND_HANDLERS[type(command)]
        start_ts = datetime.now()
        status = "Created"

        #  TODO: Convert values to upper case
        # params = {k: v.upper() for k, v in params.items() if isinstance(v, str)}
        # logger.info(f"User provided params converted to upper case: ")
        if isTrain:

            self.runConfig = pred_db.create_train_run_config(params, username)

            self.max_jobs = self.runConfig.jobs_count
            self.trainRunId = self.runConfig.train_run_id
            self.modelConfigList = pred_db.get_model_configs_by_train_id(
                self.trainRunId
            )
            self.trainingMediator = ModelTrainingMediator(
                self.runConfig, self.modelConfigList, run_id=self.trainRunId
            )
            self.runId = self.trainRunId
            pred_db.update_train_status(
                self.trainRunId, status=status, startTs=start_ts
            )

        else:
            scoreParams = params["scoring"]
            self.trainUuid = scoreParams.get("train_run_id", None)  ## TODO: use db
            scoreParams["username"] = username
            self.scoreConfig = pred_db.create_score_run_config(params)
            logger.info(
                f"------Scoring IDs: {self.scoreConfig.score_run_id}"
                f"/{self.scoreConfig.score_run_uuid}"
            )
            self.runId = self.scoreConfig.score_run_id
            self.scoreModelList = pred_db.get_score_model_summaries_by_score_id(
                self.scoreConfig.score_run_id
            )
            self.scoringMediator = ScoringMediator(
                self.scoreConfig, self.scoreModelList
            )
            pred_db.update_score_status(
                self.scoreConfig.score_run_id, status=status, startTs=start_ts
            )

    # ----------------------------------------------------------------------
    #          TRAINING AND TUNING
    # ----------------------------------------------------------------------

    def handle_command(self) -> bool:
        self._update_train_status(status="Submitted", isStart=True)
        status = False
        logger.info(f"******Task Type: {self.runConfig.train_task_type}")
        RayCommandClass = RayLocal if RUN_LOCAL else RayDistributed
        try:
            if (
                self.runConfig.train_task_type.upper()
                == PREDICTION_CONTEXT.default_training_task.upper()
            ):
                logger.info(
                    f"********Initiating model TRAINING job for {self.trainRunId}"
                )
                ray_cmd = RayCommandClass(
                    run_id=self.runConfig.train_run_id,
                    run_uuid=self.runConfig.train_run_uuid,
                    action="train",
                    username=self.runConfig.username,
                )
                status = self._exec_ray(ray_cmd, task="TRAIN/MODEL")
            elif self.runConfig.train_task_type.upper() == "TUNING":
                ray_cmd = RayCommandClass(
                    run_id=self.runConfig.train_run_id,
                    run_uuid=self.runConfig.train_run_uuid,
                    username=self.runConfig.username,
                )
                logger.info(f"**********Initiating model TUNING job: {ray_cmd}")
                status = self._exec_ray(ray_cmd, task="TUNING")

        except Exception as e:
            logger.error(f"Training failed for train_run_id: {self.trainRunId}")
            logger.exception(e)

        return status

    def _exec_ray(self, ray_command: RayCommand, task: str = "Train") -> bool:
        status = False
        try:
            # Submit Job
            result = ray_command.run()
            status = result not in ["Unknown", "Failed"]
        except Exception as e:
            logger.error(f"Failed to run ray execution task {task}")
            logger.exception(e)
            raise e
        finally:
            return status

    def _update_train_status(
        self, status: str, isStart: bool = True, fileName: str = ""
    ) -> bool:
        if isStart:
            pred_db.update_train_status(
                self.trainRunId, status=status, startTs=datetime.now()
            )
        else:
            pred_db.update_train_status(
                trainRunId=self.trainRunId,
                status=status,
                endTs=datetime.now(),
                resultPath=fileName,
            )
        return True

    # ---------------------------------SCORING/Forecast------------------------------------
    def predict(self) -> str:
        startTs = datetime.now()
        status = "Submitted"
        try:
            #  Update the database status
            pred_db.update_score_status(
                self.scoreConfig.score_run_id, status=status, startTs=startTs
            )
            logger.info(
                f"********Initiating SCORING job for {self.scoreConfig.score_run_id}"
            )
            RayCommandClass = RayLocal if RUN_LOCAL else RayDistributed
            rayCmd = RayCommandClass(
                run_id=self.scoreConfig.score_run_id,
                run_uuid=self.scoreConfig.score_run_uuid,
                entry_point="prediction.app.worker.scoring_mediator",
                action="score",
                username=self.scoreConfig.username,
            )
            if not self._exec_ray(ray_command=rayCmd, task="score"):
                status = "Failed"
            logger.info(
                f"**********Score status for {self.scoreConfig.score_run_id}: {status}"
            )
        except Exception as e:
            status = "Failed"
            # This is to update the status of the job in case any Exception is thrown
            logger.error(
                f"Exception in executing scoring for {self.scoreConfig.score_run_uuid}"
            )

            logger.exception(e)
        finally:
            #  Update the database with status-------------
            pred_db.update_score_status(
                scoreRunId=self.scoreConfig.score_run_id,
                status=status,
                endTs=datetime.now(),
            )

        return status


class PredictionJobHandler:
    def __init__(self, run_uuid: UUID, run_id: int, user: str):
        self.run_uuid = run_uuid
        self.run_id = run_id
        self.username = user
        self.job_name = f"ray-client-{self.run_id}-{self.run_uuid}"

    def status_of_job(self):
        ray_job_handler = RayJobHandler(name=self.job_name)
        status = ray_job_handler.ray_job_status()
        return status

    def delete_active_job(self, state: str, train_job: bool) -> str:
        status = "Failed"
        state_of_job = state
        try:
            if state_of_job == "Active":
                ray_delete_job_handler = RayJobHandler(self.job_name)
                delete_ray_job = ray_delete_job_handler.ray_job_delete()
                if delete_ray_job == "Success":
                    logger.info(
                        f"The job with Run ID {self.run_uuid} has been killed successfully"
                    )
                    status = "Success"
                    if train_job:
                        pred_db.update_train_status(self.run_id, status="Cancelled")
                    else:
                        pred_db.update_score_status(self.run_id, status="Cancelled")
            else:
                logger.error(
                    f"Failed to delete job as job status is not Active. It is {state_of_job}"
                )
        except Exception as e:
            logger.error(f"Exception in deleting job for {self.run_uuid}")

            logger.exception(e)
        return status
