import json
import os
from datetime import datetime
from itertools import count
from typing import Dict, List

import numpy as np
import pandas as pd
import ray
from dateutil.relativedelta import relativedelta
from loguru import logger

import prediction.src.db.prediction_database as pred_db
from prediction.app.prediction_commands import DataCommand, ScoringCommand
from prediction.app.worker.initializer import initialize
from prediction.app.worker.ray_cluster_utils import ray_cluster
from prediction.app.worker.resolvers import (
    resolve_model_command_modeler,
    resolve_model_data_preprocessor,
)
from prediction.log_config import setup_logging
from prediction.src import s3_files
from prediction.src.s3_files import read_s3_file


class ScoringMediator:
    def __init__(self, config: pred_db.ScoreRunConfig, scoreSummaries: List):
        """
        Initialize the database config
        Popualte queues:
        1. Data queue
        2. Model_Queue
        """
        self.DATA_QUEUE_ID = []
        self.MODEL = []
        self.scoreRunConfig = config
        self.score_run_id = config.score_run_id
        self.dataCommand = DataCommand(
            target_data_location=config.target_data_location,
            feature_data_location=config.feature_data_location,
            target_feature_mapping_location=config.target_feature_mapping_location,
            train_data_end_dtm=None,
            test_data_end_dtm=None,
            model_name=None,
            validation_data_end_dtm=None,
            file_type=config.score_data_file_type,
        )
        # TODO: Create ScoreCommand
        for score in scoreSummaries:
            scoreCmd = ScoringCommand(
                score_name=config.score_name,
                model_name=score.model_name,
                model_config=score.model_config,
                score_status=config.score_status,
                prediction_steps=score.prediction_steps,
                score_time_frequency=score.score_time_interval,
                target_column=score.target_column_name,
                # model_version =
                # model_location =
                score_loss_function=score.score_loss_function,
                prediction_count=score.prediction_count,
                score_description=config.score_description,
                train_run_id=config.train_run_id,
                job_name=None,
            )
            resolve_model_command_modeler(scoreCmd)
            resolve_model_data_preprocessor(scoreCmd)
            scoreCmd.db_id = score.score_model_id
            scoreCmd.job_name = pred_db.get_model_configs_by_train_id(
                scoreCmd.train_run_id
            )[0].model_job_name
            self.MODEL.append(scoreCmd)

    def _prepare_score_data(self, score_command: ScoringCommand):
        # dataHandler = DataHandler()
        data_preprocessor = score_command.data_preprocessor()
        # data_dict = dataHandler.handle(self.dataCommand)
        data_dict = data_preprocessor.handle(self.dataCommand, score_command)
        data_references = {}
        logger.info(
            f"******TrainRunID: {self.score_run_id}" "-- DATA PREP: START*************"
        )
        self.target_columns = list(data_dict)
        if score_command.model_name == "DEEPAR":
            data_references["DEEPAR"] = {
                "data": ray.put(data_dict),
            }
        else:
            for target_column, data_dict in data_dict.items():
                df = data_dict["data"]
                df_features = data_dict["features"]
                split_data = {"train": ray.put(df)}
                data_references[target_column] = {
                    "data": split_data,
                    "features": ray.put(df_features),
                }
                logger.debug(
                    f"******ScoreRunID: {self.score_run_id}"
                    f"-- DATA PREP for {target_column} DONE*************"
                )
        self.DATA_QUEUE_ID = ray.put(data_references)

    def score_in_parallel(self) -> str:
        # Step 1: Prepare scoring data
        score_status = "Failed"
        start_ts = datetime.now()
        try:
            with ray_cluster("score", os.environ.get("CLUSTER_ID", self.score_run_id)):
                logger.info("Preparing Data for Scoring")
                for model_cmd in self.MODEL:
                    self._prepare_score_data(model_cmd)

                # STEP 2: Run scoring
                logger.info("Starting Scoring on the provided data")
                start_ts = datetime.now()
                pred_db.update_score_status(
                    self.score_run_id, status="Running", startTs=start_ts
                )
                tasks = []
                for model_command in self.MODEL:
                    handler = model_command.modeler(
                        target_references_id=self.DATA_QUEUE_ID
                    )
                    if model_command.model_name == "DEEPAR":
                        handler.load_data(model_command, self.dataCommand)

                    for target in self.target_columns:
                        scorer = ParallelScore.remote()
                        tasks.append(
                            scorer.score_single_target.remote(
                                handler, model_command, target
                            )
                        )

                logger.info("Waiting for tasks")
                total = len(tasks)
                completed = count(1)
                while tasks:
                    ready, tasks = ray.wait(tasks)
                    logger.debug(f"Completed {next(completed)}/{total}")
                    result = ray.get(ready)
                    logger.info(f"{result=}")
                    if result:
                        try:
                            pred_db.create_score_model_summary(**result[0])
                        except Exception as e:
                            logger.error("Exception occured while saving score results")
                            logger.exception(e)
                    else:
                        logger.error("No scoring result returned")

                score_status = "Completed"
        except Exception as e:
            logger.error("Exception occured while scoring")
            logger.exception(e)
        finally:
            logger.info(f"Setting {score_status=}")
            pred_db.update_score_status(
                self.score_run_id, status=score_status, startTs=start_ts
            )
        return score_status


@ray.remote(max_restarts=-1, max_task_retries=-1)
class ParallelScore:
    def __init__(self):
        self.status = None
        setup_logging()

    def get_status(self):
        return self.status

    def score_single_target(self, handler, model_command, column) -> Dict:
        model_name = model_command.model_name
        with logger.contextualize(model_name=model_name, column=column):
            try:
                # logger.info(cmd)
                status = "Failed"
                model_start_ts = datetime.now()
                forecast = handler.handle(model_command, target=column)
                if model_name == "DEEPAR":
                    if forecast.isnull().values.any():
                        forecast = []
                        logger.info(
                            "Forecast has errored out and returned NaN as the forecast values"
                        )
                    else:
                        forecast["datetime"] = forecast["datetime"].dt.strftime(
                            "%Y-%m-%d"
                        )
                        forecast = forecast.to_json()
                else:
                    if not isinstance(forecast, List):
                        forecast = [forecast]
                    if np.isnan(forecast).all():
                        forecast = []
                        logger.info(
                            "Forecast has errored out and returned NaN as the forecast values"
                        )
                if forecast:
                    status = "Completed"
                logger.info(
                    f"Creating ScoreModelSummary for {model_name}-{column} Forecast: {forecast}"
                )
                self.status = status
                return {
                    "scoreModelId": model_command.db_id,
                    "col": column,
                    "status": status,
                    "forecastedQuantity": forecast,
                    "startTs": model_start_ts,
                    "endTs": datetime.now(),
                }
            except Exception as e:
                # TODO
                self.status = "Failed"
                logger.info(f"Error: Failed to score {column}")
                logger.exception(e)
                return {}


def create_score_result_df(
    score_run_id: int, target_data_location: str
) -> pd.DataFrame:
    #  Generate data for saving to S3
    score_results = pred_db.get_score_model_summaries_by_score_id(score_run_id)
    #  Generate forecast dates
    scoreSummary = score_results[0]
    predSteps = scoreSummary.prediction_steps
    predCount = scoreSummary.prediction_count
    target_data_location = target_data_location
    targetDf = read_s3_file(target_data_location)
    maxIndex = targetDf.index.max()
    toAdd = (predSteps - predCount) + 1 if predSteps > predCount else 1
    #  If the prediction is for entire predSteps then we need to add one as
    #  prediction starts from next month and range function is exclusive for the end
    endIndex = predSteps + 1
    datesList = [maxIndex + relativedelta(months=+i) for i in range(toAdd, endIndex)]
    forecastDates = pd.DatetimeIndex(datesList)
    logger.info(f"# of forecast dates: {len(forecastDates)}")
    #  Create data dictionary of forecast results
    scoreDict = {}
    scoreDict["forecast_date"] = forecastDates

    for scoreSummary in score_results:
        columnName = scoreSummary.target_column_name
        forecastVolume = scoreSummary.forecasted_quantity

        if scoreSummary.model_name == "DEEPAR":
            if scoreSummary.score_trial_status == "Completed":
                forecastVolume = json.loads(forecastVolume)
                # for k, v in forecastVolume["index_cols"].items():
                #     if v[0] == columnName:
                #         col_index = k
                # forecastVolume["prediction_50"][col_index]
                # forecastVolume = forecastVolume["prediction_50"][col_index]
                # scoreDict[columnName] = forecastVolume
                scoreDict = forecastVolume
                # NOTE: Writing the raw results, DS refers to other quantiles as well.
            else:
                continue
        else:
            if columnName not in scoreDict and len(forecastVolume) == predCount:
                scoreDict[columnName] = forecastVolume
    #  Create dataframe from the dictionary

    try:
        df = pd.DataFrame.from_dict(scoreDict, orient="columns")
        return df
    except Exception as e:
        logger.error("Error in creating df")
        logger.error(
            "******# of forecast dates of length "
            f"{len(forecastDates)}: {forecastDates}"
        )
        logger.error(f"*******Error in creating df for score results: {scoreDict}")
        logger.exception(e)
        raise e


def write_score_results(file_name: str, score_run_id: int, target_data_location: str):
    try:
        #  Produce score results
        df = create_score_result_df(score_run_id, target_data_location)
        s3_files.save_train_result(df=df, s3Path=file_name)
        return file_name
    except Exception as e:
        logger.error("Error in saving prediction results to S3")
        logger.exception(e)
        raise e


def _dispatch_tasks(args):
    run_id = args.run_id
    action = args.action
    with logger.contextualize(run_id=run_id, action=action):
        logger.info(
            f"-------------------------------INTIATING {action}------------------"
        )
        logger.info(f"********Scoring runID: {run_id}")
        logger.info(f"********Scoring action: {action}")
        status = "Failed"
        try:
            logger.info(f"*********Parallel {action} for runID = {run_id}")
            score_run_config = pred_db.get_score_run_config(run_id)
            logger.info(f"*********Model {action} run config: {score_run_config}")
            score_configs = pred_db.get_score_model_summaries_by_score_id(
                score_run_config.score_run_id
            )
            logger.info(f"**********Score configs: {score_configs}")
            logger.info(f"**********Total # of scores: {len(score_configs)}")
            mediator = ScoringMediator(score_run_config, score_configs)
            logger.info("*********Calling SCORING in Parallel")
            status = mediator.score_in_parallel()

            #  Create file name
            prefix = f"{score_run_config.score_name}_{score_run_config.score_run_uuid}"
            file_name = s3_files.create_s3_path(
                s3Folder=score_run_config.score_data_location,
                trainName=prefix,
                trainTs=score_run_config.score_start_ts,
            )

            write_score_results(
                file_name, run_id, score_run_config.target_data_location
            )
            pred_db.update_score_status(run_id, status=status, resultPath=file_name)
        except Exception as e:
            pred_db.update_score_status(run_id, status="Failed")
            logger.error(
                f"***********Exception in {action} using Ray for runId: {run_id}"
            )
            logger.exception(e)
        logger.info(
            "-------------------------------COMPLETED Scoring------------------"
        )


if __name__ == "__main__":
    args = initialize(actions=["score"])
    with logger.contextualize(user=args.user, run_id=args.run_id):
        logger.debug(f"*********Training arguments: {args} Calling _dispatch_tasks")
        _dispatch_tasks(args)
