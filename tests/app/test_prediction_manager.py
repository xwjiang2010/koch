"""
    File name: test_prediction_manager.py
    Author: Badrul Islam
    Python Version: 3+
"""

import copy
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import unittest
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import ray

import prediction.app.prediction_manager as pred_manager
import prediction.src.db.prediction_database as pred_db
from prediction.app.prediction_commands import ScoringCommand
from prediction.src.static import PREDICTION_CONTEXT

logger = logging.getLogger(PREDICTION_CONTEXT.logger_name)

S3_PRED_PATH = "tests/resources/data/"


@pytest.fixture
def trainDict():
    return {
        "master": {
            "exec_environment": "sequential",
            "data_storage_type": "s3",
            "data_file_type": "csv",
        },
        "training": {
            "train_name": "model-mediator-test",
            "train_description": "Testing training of ARIMA model",
            "train_task_type": "Model",
            "train_job_type": "Single-Target",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "feature_data_location": "",
            "model_version": "pred-manager-test",
            "data_version": "v1.0",
            "loss_function": "MAPE",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-02-05",
            "time_interval": "D",
            "model_names": ["ARIMA"],
            "prediction_steps": "0",
        },
    }


@pytest.fixture
def modelDict():
    return {
        "ARIMA": {
            "model_name": "ARIMA",
            "model_time_interval": "D",
            "model_config": {
                "parameters": {"p": 1, "d": 2, "q": 0},
                "hyperparameters": {"disp": 0},
            },
        }
    }


def testPredictionManagerCreationTrainMin():
    trainingDict = {
        "training": {
            "train_name": "model-minimum-config-jupyter",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
            "prediction_steps": "0",
        }
    }
    mgr = pred_manager.PredictionMediator(trainingDict, username="devin")
    assert mgr is not None
    assert mgr.runConfig is not None
    assert mgr.modelConfigList is not None
    assert len(mgr.modelConfigList) > 0


def testPredictionManagerCreation(trainDict, modelDict):
    trainingDict = {
        "master": {  ##OPTIONAL
            "exec_environment": "sequential",
            "data_storage_type": "s3",
            "data_file_type": "csv",
        },
        "training": {
            "train_name": "model-mediator-test",  ##REQ
            "train_description": "Testing training of ARIMA model",
            "train_task_type": "TUNING",
            "train_job_type": "Single-Target",
            "target_data_location": "tests/resources/data/test_single_target.csv",  ##REQ
            "feature_data_location": "",
            "model_version": "pred-manager-test",  ##train_name-version-[data-version]-YYYYYMMDDHH:MM
            "data_version": "v1.0",
            "loss_function": "MAPE",
            "train_data_end_dtm": "1990-01-31",  ##REQ
            "test_data_end_dtm": "1990-02-03",  ##REQ
            "validation_data_end_dtm": "1990-02-05",  ##REQ
            "time_interval": "D",  ##REQ
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],  ##REQ
            "tuning_algorithm": "GRID-SEARCH",
        },
        "models": {
            "ARIMA": {
                "model_name": "ARIMA",
                "model_time_interval": "M",
                "model_config": {
                    "parameters": {
                        "p": [1 - 7],
                        "d": [1 - 2],
                        "q": [0 - 3],
                        "hyperparameters": {"disp": 0},
                    },
                },
            }
        },
    }

    trainDict["models"] = modelDict
    mgr = pred_manager.PredictionMediator(trainDict, username="devin")
    assert mgr is not None
    assert mgr.runConfig is not None
    assert mgr.modelConfigList is not None
    assert len(mgr.modelConfigList) > 0


def testPredictionManagerTrainSingle(trainDict, modelDict):
    trainDict["models"] = modelDict
    mgr = pred_manager.PredictionMediator(trainDict, username="devin")
    assert mgr is not None
    assert mgr.runConfig is not None
    assert mgr.modelConfigList is not None
    assert len(mgr.modelConfigList) > 0
    result = mgr.handle_command()
    logger.info(f"Training result: {result}")
    assert result is not None


@pytest.mark.slow
def testPredictionManagerTrainMultiple():
    """trainDict2 = {
        "training": {
            "train_name": "model-minimum-config-jupyter",
            "target_data_location": "s3://kbs-analytics-dev-c3-public/fhr-public/lv-polypropylene-demand-forecast/input/Training/TransformedData/test/TrainingInputData_Transformed_Test_Sample_Full_History_1.csv", ##tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "2020-01-01", #"1990-01-31",
            "test_data_end_dtm": "2020-09-01", #"1990-02-03",
            "validation_data_end_dtm": "2020-12-01", #"1990-03-02",
            "model_names": ["HOLTWINTERS"] ##["ARIMA", "HOLTWINTERS", "PROPHET"],
        }
    }"""
    trainDict2 = {
        "training": {
            "train_name": "model-minimum-config-jupyter",
            "target_data_location": "s3://kbs-analytics-dev-c3-public/fhr-public/lv-polypropylene-demand-forecast/input/Training/TransformedData/test/TrainingInputData_Transformed_Test_Sample_Full_History_1.csv",  ##tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
        }
    }
    mgr = pred_manager.PredictionMediator(trainDict2, username="devin")
    assert mgr is not None
    assert mgr.runConfig is not None
    assert mgr.modelConfigList is not None
    assert len(mgr.modelConfigList) > 0
    assert mgr.trainingMediator is not None
    result = mgr.handle_command()
    logger.info(f"Training result: {result}")
    assert result is not None
    assert result == "Completed"


@pytest.mark.ray
def testPredictionManagerTune():
    trainDict2 = {
        "training": {
            "train_name": "model-minimum-config-jupyter",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": ["ENSEMBLE"],
            "metrics": [{"name": "bias"}],
        }
    }

    mgr = pred_manager.PredictionMediator(trainDict2, username="test")
    assert mgr is not None
    assert mgr.runConfig is not None
    assert mgr.runConfig.model_type is not None
    assert mgr.runConfig.model_type.upper() == "ENSEMBLE"

    assert mgr.modelConfigList is not None
    assert len(mgr.modelConfigList) > 0
    assert mgr.modelConfigList[0].metrices == {"bias": True}
    assert mgr.trainingMediator is not None
    result = mgr.handle_command()
    logger.info(f"Training result: {result}")
    assert result is not None


@pytest.mark.skip
def testTuneMultiplePods():
    tuneDict = {
        "master": {
            "exec_environment": "Parallel",
            "data_storage_type": "s3",
            "data_file_type": "csv",
            "resources": {"max_parallel_pods": 45},
        },
        "training": {
            "train_task_type": "Tuning",
            "train_name": "Tuning-Maxim-fhr-lv-pilot-data",
            "target_data_location": "s3://prediction-services/data/fhr_lv_pilot_data.csv",
            "score_data_location": "s3://prediction-services/score/",
            "train_data_end_dtm": "2020-08-01",
            "test_data_end_dtm": "2020-11-01",
            "validation_data_end_dtm": "2021-02-01",
            "model_names": ["ENSEMBLE"],
        },
    }
    mgr = pred_manager.PredictionMediator(tuneDict)
    assert mgr is not None
    assert mgr.runConfig is not None
    assert mgr.runConfig.model_type is not None
    assert mgr.runConfig.model_type.upper() == "ENSEMBLE"
    assert mgr.max_jobs == tuneDict["master"]["resources"]["max_parallel_pods"]
    assert mgr.modelConfigList is not None
    assert len(mgr.modelConfigList) > 0

    result = mgr.handle_command()
    logger.info(f"Training result: {result}")
    assert len(result) > 0


##-----------------------TESTING TUNE---------------------------------


@pytest.mark.skip
def testPredictionManagerTune(trainDict, modelDict):
    tuneDict = trainDict
    tuneDict["training"]["train_name"] = "TESTING-RAY-TUNE"
    tuneDict["training"]["train_task_type"] = "TUNING"
    tuneDict["models"] = modelDict
    mgr = pred_manager.PredictionMediator(tuneDict)
    assert mgr is not None
    assert mgr.runConfig is not None
    assert mgr.modelConfigList is not None
    assert len(mgr.modelConfigList) > 0
    assert mgr.trainingMediator is not None
    result = mgr.handle_command()
    logger.info(f"Training result: {result}")
    assert result is None


##----------------------SCORING TESTS-----------------------------
@pytest.fixture
def scoreConfigDict():
    return {
        "master": {
            "exec_environment": "sequential",
            "data_storage_type": "s3",
            "data_file_type": "csv",
        },
        "scoring": {
            "train_name": "test-arima",
            "train_description": "Testing insertion into the table 2",
            "train_task_type": "Model",
            "train_job_type": "Single-Target",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "feature_data_location": "",
            "model_version": "abc",
            "data_version": "v1",
            "loss_function": "MAPE",
            "train_data_end_dtm": "2010-10-4",
            "test_data_end_dtm": "2011-01-01",
            "validation_data_end_dtm": "2012-01-01",
            "time_interval": "D",
            "model_names": ["ARIMA"],
            "prediction_steps": 3,
            "prediction_count": 2,
            "username": "devin",
        },
    }


@pytest.fixture
def scoreModelConfigDict():
    return {
        "ARIMA": {
            "model_name": "ARIMA",
            "model_time_interval": "M",
            "model_config": {
                "parameters": {"p": 1, "d": 1, "q": 0},
                "hyperparameters": {"disp": 0},
            },
        }
    }


def testPredictionManagerScoreMin():
    trainDict = {
        "training": {
            "train_name": "model-minimum-config",
            "target_data_location": "s3://kbs-analytics-dev-c3-public/fhr-public/lv-polypropylene-demand-forecast/input/Training/TransformedData/test/TrainingInputData_Transformed_Test_Sample_Full_History_1.csv",  ##tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": [
                "DEEPAR",
            ],
            "prediction_steps": 1,
        }
    }
    mgr = pred_manager.PredictionMediator(trainDict, username="devin")
    train_run_id = mgr.runConfig.train_run_uuid

    scoringDict = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["DEEPAR"],
            "prediction_steps": 3,
            "username": "neelesh",
            "train_run_id": train_run_id,
        }
    }
    mgr = pred_manager.PredictionMediator(scoringDict, isTrain=False, username="devin")
    assert mgr is not None
    assert mgr.scoreConfig is not None
    assert mgr.scoreModelList is not None
    assert len(mgr.scoreModelList) > 0
    assert mgr.scoringMediator is not None


def testPredictionManagerCreationScoreMin():

    trainDict = {
        "training": {
            "train_name": "model-minimum-config",
            "target_data_location": "s3://kbs-analytics-dev-c3-public/fhr-public/lv-polypropylene-demand-forecast/input/Training/TransformedData/test/TrainingInputData_Transformed_Test_Sample_Full_History_1.csv",  ##tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": [
                "ARIMA",
            ],
            "prediction_steps": 1,
        }
    }
    mgr = pred_manager.PredictionMediator(trainDict, username="devin")
    train_run_id = mgr.runConfig.train_run_uuid

    scoringDict = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["ARIMA"],
            "prediction_steps": 3,
            "username": "neelesh",
            "train_run_id": train_run_id,
        }
    }
    mgr = pred_manager.PredictionMediator(scoringDict, isTrain=False, username="devin")
    assert mgr is not None
    assert mgr.scoreConfig is not None
    assert mgr.scoreModelList is not None
    assert len(mgr.scoreModelList) > 0
    assert mgr.scoringMediator is not None
    scoreResult = mgr.predict()
    logger.info(f"ScoreResult: {scoreResult}")
    assert scoreResult is not None
    assert len(scoreResult) > 0
    ## TODO assert scoreResult == "Completed"


def testPredictionManagerScoreStepsCount():

    trainDict = {
        "training": {
            "train_name": "model-minimum-config",
            "target_data_location": "s3://kbs-analytics-dev-c3-public/fhr-public/lv-polypropylene-demand-forecast/input/Training/TransformedData/test/TrainingInputData_Transformed_Test_Sample_Full_History_1.csv",  ##tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": [
                "ARIMA",
            ],
            "prediction_steps": 1,
        }
    }
    mgr = pred_manager.PredictionMediator(trainDict, username="devin")
    train_run_id = mgr.runConfig.train_run_uuid

    scoringDict = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["ARIMA", "HOLTWINTERS"],
            "prediction_steps": 12,
            "prediction_count": 10,
            "username": "devin",
            "train_run_id": train_run_id,
        }
    }
    mgr = pred_manager.PredictionMediator(scoringDict, isTrain=False, username="devin")
    assert mgr is not None
    assert mgr.scoreConfig is not None
    assert mgr.scoreModelList is not None
    assert len(mgr.scoreModelList) > 0
    assert mgr.scoringMediator is not None
    scoreResult = mgr.predict()
    logger.info(f"ScoreResult: {scoreResult}")
    assert scoreResult is not None
    assert len(scoreResult) > 0
    ## TODO assert scoreResult == "Completed"


def testPredictionManagerScoreMinPrevTrain():
    ## Create database record for training
    trainConfigDict = {
        "training": {
            "train_name": "test-pred-manager-score-prev",
            "target_data_location": "tests/resources/data/h0500hn_ft-worth.csv",
            "train_data_end_dtm": "2019-12-01",
            "test_data_end_dtm": "2020-06-01",
            "validation_data_end_dtm": "2020-11-01",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
            "prediction_steps": "0",
            "train_run_id": 1,
        }
    }
    config = pred_db.create_train_run_config(trainConfigDict, username="devin")
    ## ScoreRunConfig--------------------------------------
    trainRunUuid = config.train_run_uuid

    scoringDict = {
        "scoring": {
            "score_name": "test-pred-manager-score-prev",
            "target_data_location": "tests/resources/data/h0500hn_ft-worth.csv",
            "train_run_id": trainRunUuid,
            "prediction_steps": 12,
            "prediction_count": 10,
            "username": "devin",
        }
    }
    mgr = pred_manager.PredictionMediator(scoringDict, isTrain=False, username="devin")
    assert mgr is not None
    scoreConfig = mgr.scoreConfig
    assert scoreConfig is not None
    scoreId = scoreConfig.score_run_id
    assert scoreId is not None
    assert mgr.scoreModelList is not None
    assert len(mgr.scoreModelList) > 0
    assert mgr.scoringMediator is not None

    scoreStatus = mgr.predict()
    logger.info(f"scoreStatus: {scoreStatus}")
    assert scoreStatus is not None
    assert len(scoreStatus) > 0
    ## TODO assert scoreStatus == "Completed"

    scoreResult = pred_db.get_score_run_config(scoreId)
    logger.info(f"scoreResult: {scoreResult}")
    assert scoreResult is not None


def testPredictionManagerScoreHoltMultipleTS():

    train_dict = {
        "training": {
            "train_name": "model-minimum-config",
            "target_data_location": "s3://kbs-analytics-dev-c3-public/fhr-public/lv-polypropylene-demand-forecast/input/Training/TransformedData/test/TrainingInputData_Transformed_Test_Sample_Full_History_1.csv",  ##tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": [
                "ARIMA",
            ],
            "prediction_steps": 1,
        }
    }
    mgr = pred_manager.PredictionMediator(train_dict, username="devin")
    train_run_id = mgr.runConfig.train_run_uuid

    targetDataLoc = (
        "tests/resources/data/TrainingInputData_Transformed_Test_Sample_2.csv"
    )
    scoringDict = {
        "scoring": {
            "score_name": "test-pred-manager-score-multiplets",
            "target_data_location": targetDataLoc,
            "model_names": ["HOLTWINTERS"],
            "prediction_steps": 12,
            "prediction_count": 10,
            "train_run_id": train_run_id,
            "username": "devin",
        }
    }
    mgr = pred_manager.PredictionMediator(scoringDict, isTrain=False, username="devin")
    assert mgr is not None
    scoreConfig = mgr.scoreConfig
    assert scoreConfig is not None
    scoreId = scoreConfig.score_run_id
    assert scoreId is not None
    assert mgr.scoreModelList is not None
    assert len(mgr.scoreModelList) > 0
    assert mgr.scoringMediator is not None

    scoreStatus = mgr.predict()
    logger.info(f"scoreStatus: {scoreStatus}")
    assert scoreStatus is not None
    assert len(scoreStatus) > 0
    ## TODO assert scoreStatus == "Completed with Failure"

    scoreResult = pred_db.get_score_run_config(scoreId)
    logger.info(f"scoreResult: {scoreResult}")
    assert scoreResult is not None


def testPredictionManagerArimaSingleTarget():
    scoringDict = {
        "scoring": {
            "score_name": "c3-predictions",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["ARIMA"],
            "prediction_steps": 12,
            "prediction_count": 12,
            "train_run_id": "f6c4164f-f356-4baf-826e-2e6ec155864d",
            "username": "devin",
        }
    }
    mgr = pred_manager.PredictionMediator(scoringDict, isTrain=False, username="devin")
    assert mgr is not None
    scoreConfig = mgr.scoreConfig
    assert scoreConfig is not None
    scoreId = scoreConfig.score_run_id
    assert scoreId is not None
    assert mgr.scoreModelList is not None
    assert len(mgr.scoreModelList) > 0
    assert mgr.scoringMediator is not None

    scoreStatus = mgr.predict()
    logger.info(f"scoreStatus: {scoreStatus}")
    assert scoreStatus is not None
    assert len(scoreStatus) > 0
    ##TODO: Need to mock Ray assert scoreStatus == "Completed"

    scoreResult = pred_db.get_score_run_config(scoreId)
    logger.info(f"scoreResult: {scoreResult}")
    assert scoreResult is not None


def testPredictionManagerEnsembleSingleTarget():

    train_dict = {
        "training": {
            "train_name": "model-minimum-config",
            "target_data_location": "s3://kbs-analytics-dev-c3-public/fhr-public/lv-polypropylene-demand-forecast/input/Training/TransformedData/test/TrainingInputData_Transformed_Test_Sample_Full_History_1.csv",  ##tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": [
                "ARIMA",
            ],
            "prediction_steps": 1,
        }
    }
    mgr = pred_manager.PredictionMediator(train_dict, username="devin")
    train_run_id = mgr.runConfig.train_run_uuid

    scoringDict = {
        "scoring": {
            "score_name": "c3-predictions-ENSEMBLE",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["ENSEMBLE"],
            "prediction_steps": 12,
            "prediction_count": 12,
            "username": "devin",
            "train_run_id": train_run_id,
        }
    }
    mgr = pred_manager.PredictionMediator(scoringDict, isTrain=False, username="devin")
    assert mgr is not None
    scoreConfig = mgr.scoreConfig
    assert scoreConfig is not None
    scoreId = scoreConfig.score_run_id
    assert scoreId is not None
    assert scoreConfig.model_type is not None
    assert scoreConfig.model_type.upper() == "ENSEMBLE"
    assert mgr.scoreModelList is not None
    assert len(mgr.scoreModelList) > 0
    assert mgr.scoringMediator is not None

    scoreStatus = mgr.predict()
    logger.info(f"scoreStatus: {scoreStatus}")
    assert scoreStatus is not None
    assert len(scoreStatus) > 0
    ## TODO assert scoreStatus == "Completed"

    scoreResult = pred_db.get_score_run_config(scoreId)
    logger.info(f"scoreResult: {scoreResult}")
    assert scoreResult is not None


@pytest.mark.slow
def testPredictionManagerEnsemblePrevFHR():
    scoringDict = {
        "scoring": {
            "score_name": "postman-test-score-prev-train",
            "target_data_location": "tests/resources/data/TrainingInputData_Transformed_Test.csv",
            "model_names": ["ENSEMBLE"],
            "prediction_steps": 12,
            "prediction_count": 10,
            "train_run_id": "72ee0e3a-84fb-4b4e-bd59-09e59f8a209b",
        }
    }

    mgr = pred_manager.PredictionMediator(scoringDict, isTrain=False, username="devin")
    assert mgr is not None
    scoreConfig = mgr.scoreConfig
    assert scoreConfig is not None
    scoreId = scoreConfig.score_run_id
    assert scoreId is not None
    assert scoreConfig.model_type is not None
    assert scoreConfig.model_type.upper() == "ENSEMBLE"
    assert mgr.scoreModelList is not None
    assert len(mgr.scoreModelList) > 0
    assert mgr.scoringMediator is not None

    scoreStatus = mgr.predict()
    logger.info(f"scoreStatus: {scoreStatus}")
    assert scoreStatus is not None
    assert len(scoreStatus) > 0
    assert scoreStatus == "Completed"

    scoreResult = pred_db.get_score_run_config(scoreId)
    logger.info(f"scoreResult: {scoreResult}")
    assert scoreResult is not None


##-------------------------Ray Train ---------------------
@pytest.mark.skip
def testPredictionManagerTrainingParallel(tempdir):
    molex = {
        "training": {
            "train_name": "molex-APS-SKU-CUST",
            "train_description": "PredictionManager test - Molex APS data training for ARIMA, HOLT, and PROPHET",
            "target_data_location": f"{S3_PRED_PATH}/proc_SKU_CUST_APS_remove-zeros.csv",  ##TrainingInputData_Transformed_Test.csv",  ##TrainingInputData_Transformed_Test_Sample_2.csv",
            "train_data_end_dtm": "2020-04-01",
            "test_data_end_dtm": "2020-07-01",
            "validation_data_end_dtm": "2021-01-01",
            ##"train_data_end_dtm": "2020-1-1",
            ##"test_data_end_dtm": "2020-6-1",
            ##"validation_data_end_dtm": "2020-12-1",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
        }
    }
    mgr = pred_manager.PredictionMediator(molex, "devin")
    assert mgr is not None
    assert mgr.runConfig is not None
    assert mgr.modelConfigList is not None
    assert len(mgr.modelConfigList) > 0
    assert mgr.trainingMediator is not None

    trainingMediator = mgr.trainingMediator
    assert trainingMediator is not None
    dataCommand = trainingMediator.dataCommand
    assert dataCommand is not None
    assert dataCommand.target_data_location is not None
    ##assert "TrainingInputData_Transformed_Test.csv" in dataCommand.target_data_location
    assert dataCommand.feature_data_location == ""
    assert (
        trainingMediator.trainConfig.score_data_location
        == PREDICTION_CONTEXT.S3_TRAIN_RESULT_PATH
    )
    assert trainingMediator.MODEL_QUEUE is not None
    assert len(trainingMediator.MODEL_QUEUE) > 0

    ## Testing PARALLEL training
    ##ray.init(ignore_reinit_error=True)
    ray.init(local_mode=True, num_cpus=mp.cpu_count())
    tic = time.perf_counter()
    result = mgr.handle_command()
    toc = time.perf_counter()
    logger.info(
        f"*********Result of MOLEX training: {result}, time elapsed: {toc-tic:0.4f} seconds"
    )
    assert len(trainingMediator.DATA_QUEUE) > 1
    assert result is None

    ray.shutdown()


##---------------TESTING DATABASE ENTRIES-----------------
def test_exception_handling():
    train_dict = {
        "training": {
            "train_name": "model-minimum-config",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": [
                "ENSEMBLE",
            ],
            "prediction_steps": 1,
        }
    }
    mgr = pred_manager.PredictionMediator(train_dict, username="vaisakh")
    train_run_id = mgr.runConfig.train_run_uuid

    scoringDict = {
        "scoring": {
            "score_name": "c3-predictions-ENSEMBLE",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["ENSEMBLE"],
            "prediction_steps": 12,
            "prediction_count": 12,
            "username": "vaisakh",
            "train_run_id": train_run_id,
        }
    }
    # Mock the _exec_ray function to raise an error
    with unittest.mock.patch(
        "prediction.app.prediction_manager.PredictionMediator._exec_ray",
        side_effect=ValueError,
    ):
        mgr = pred_manager.PredictionMediator(
            scoringDict, isTrain=False, username="vaisakh"
        )
        assert mgr is not None
        scoreId = mgr.scoreConfig.score_run_id
        assert scoreId is not None
        scoreResult = mgr.predict()
        assert scoreResult is not None
        scoreReturn = pred_db.get_score_run_config(scoreId)
        assert scoreReturn is not None
        scoreStatus = scoreReturn.score_status
        assert scoreStatus == "Failed"


def test_no_exception():

    train_dict = {
        "training": {
            "train_name": "model-minimum-config",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": [
                "ENSEMBLE",
            ],
            "prediction_steps": 1,
        }
    }
    mgr = pred_manager.PredictionMediator(train_dict, username="vaisakh")
    train_run_id = mgr.runConfig.train_run_uuid
    scoringDict = {
        "scoring": {
            "score_name": "c3-predictions-ENSEMBLE",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["ENSEMBLE"],
            "prediction_steps": 12,
            "prediction_count": 12,
            "username": "vaisakh",
            "train_run_id": train_run_id,
        }
    }
    # Mock the _exec_ray function to raise an error
    with unittest.mock.patch(
        "prediction.app.prediction_manager.PredictionMediator._exec_ray",
        return_value=True,
    ):
        mgr = pred_manager.PredictionMediator(
            scoringDict, isTrain=False, username="vaisakh"
        )
        assert mgr is not None
        scoreId = mgr.scoreConfig.score_run_id
        assert scoreId is not None
        scoreResult = mgr.predict()
        assert scoreResult is not None
        scoreReturn = pred_db.get_score_run_config(scoreId)
        assert scoreReturn is not None
        scoreStatus = scoreReturn.score_status
        assert scoreStatus == "Submitted"


##-------------Testing Prediction Job Handler-------------
def test_job_handler():
    mgr = pred_manager.PredictionJobHandler(
        "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11", 1, "vaisakh"
    )
    assert mgr is not None
    run_uuid = mgr.run_uuid
    assert run_uuid is not None
    run_id = mgr.run_id
    assert run_id is not None
    job_name = mgr.job_name
    assert job_name is not None
    assert job_name == (f"ray-client-{run_id}-{run_uuid}")


def test_status_of_job():
    with unittest.mock.patch(
        "prediction.app.prediction_manager.RayJobHandler.ray_job_status",
        return_value="Active",
    ):
        mgr = pred_manager.PredictionJobHandler(
            "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11", 1, "vaisakh"
        )
        assert mgr is not None
        assert mgr.status_of_job() == "Active"

    with unittest.mock.patch(
        "prediction.app.prediction_manager.RayJobHandler.ray_job_status",
        return_value="Completed",
    ):
        mgr = pred_manager.PredictionJobHandler(
            "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11", 1, "vaisakh"
        )
        assert mgr is not None
        assert mgr.status_of_job() == "Completed"


def test_delete_active_job():
    with unittest.mock.patch(
        "prediction.app.prediction_manager.RayJobHandler.ray_job_delete",
        return_value="Unsuccessfull",
    ):
        mgr = pred_manager.PredictionJobHandler(
            "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11", 1, "vaisakh"
        )
        assert mgr is not None
        kill_result = mgr.delete_active_job("Active", True)
        assert kill_result == "Failed"

    with unittest.mock.patch(
        "prediction.app.prediction_manager.RayJobHandler.ray_job_delete",
        return_value="Success",
    ):
        mgr = pred_manager.PredictionJobHandler(
            "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11", 1, "vaisakh"
        )
        assert mgr is not None
        kill_result = mgr.delete_active_job("Active", True)
        assert kill_result == "Success"
        kill_result_non_active = mgr.delete_active_job("Completed", True)
        assert kill_result_non_active == "Failed"


def test_delete_error():
    with unittest.mock.patch(
        "prediction.app.prediction_manager.RayJobHandler.ray_job_delete",
        side_effect=ValueError,
    ):
        mgr = pred_manager.PredictionJobHandler(
            "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11", 1, "vaisakh"
        )
        assert mgr is not None
        kill_result = mgr.delete_active_job("Active", True)
        assert kill_result == "Failed"


def test_delete_active_prediction_job():
    with unittest.mock.patch(
        "prediction.app.prediction_manager.RayJobHandler.ray_job_delete",
        return_value="Unsuccessfull",
    ):
        mgr = pred_manager.PredictionJobHandler(
            "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11", 1, "vaisakh"
        )
        assert mgr is not None
        kill_result = mgr.delete_active_job("Active", False)
        assert kill_result == "Failed"

    with unittest.mock.patch(
        "prediction.app.prediction_manager.RayJobHandler.ray_job_delete",
        return_value="Success",
    ):
        mgr = pred_manager.PredictionJobHandler(
            "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11", 1, "vaisakh"
        )
        assert mgr is not None
        kill_result = mgr.delete_active_job("Active", False)
        assert kill_result == "Success"
        kill_result_non_active = mgr.delete_active_job("Completed", False)
        assert kill_result_non_active == "Failed"
