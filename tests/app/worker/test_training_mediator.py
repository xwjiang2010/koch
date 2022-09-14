import logging
import multiprocessing as mp
import time
from itertools import count
from typing import Dict
from unittest.mock import patch

import pandas as pd
import pytest
import ray
from dateutil.relativedelta import *
from pony.orm import *
from tabulate import tabulate

# import prediction.src.tune.ray_tune_utils as ray_tune
import prediction.app.worker.training_mediator as training_mediator
import prediction.src.db.prediction_database as pred_db
from prediction.app.prediction_commands import DataCommand, TrainingModel
from prediction.app.worker.data_preprocessor.tsa import TSADataPreProcessor
from prediction.src.db.prediction_database import get_train_run_config
from prediction.src.static import PREDICTION_CONTEXT

logger = logging.getLogger(PREDICTION_CONTEXT.logger_name)

S3_PRED_PATH = "tests/resources/data/"


@pytest.fixture
def ensemble_tune_config():
    return {
        "master": {
            "exec_environment": "Parallel",
            "data_storage_type": "s3",
            "data_file_type": "csv",
        },
        "training": {
            "train_task_type": "Tuning",
            "train_name": "KGSA Prediction Service Test",
            "target_data_location": "s3://kbs-analytics-dev-c3-public/fhr-public/lv-polypropylene-demand-forecast/input/Training/TransformedData/test/TrainingInputData_Transformed_Test_Sample_Full_History_1.csv",
            "score_data_location": "s3://kbs-analytics-dev-c3-public/fhr-public/lv-polypropylene-demand-forecast/output/Training/test/",
            "train_data_end_dtm": "2020-06-01",
            "test_data_end_dtm": "2020-09-01",
            "validation_data_end_dtm": "2020-12-01",
            "model_names": ["ENSEMBLE"],
            "prediction_steps": "0",
        },
    }


def testModelTrainingMediatorCreateMinimum():
    trainRunId = 143
    runConfig = get_train_run_config(trainRunId)
    assert runConfig is not None
    modelConfigs = pred_db.get_model_configs_by_train_id(trainRunId)
    assert modelConfigs is not None

    trainingMediator = training_mediator.ModelTrainingMediator(
        config=runConfig, modelConfigs=modelConfigs, run_id=trainRunId
    )
    assert trainingMediator is not None
    assert len(trainingMediator.MODEL_QUEUE) == 3
    currModelConfig = trainingMediator.MODEL_QUEUE[0]
    assert currModelConfig is not None
    assert currModelConfig.model_name in ["ARIMA", "HOLTWINTERS", "PROPHET"]
    assert currModelConfig.model_metric == "MAPE"


def testModelTrainingMediatorCreate():
    trainRunId = 16
    runConfig = get_train_run_config(trainRunId)
    assert runConfig is not None
    modelConfigs = pred_db.get_model_configs_by_train_id(trainRunId)

    assert modelConfigs is not None
    # assert isinstance(modelConfigs[0].metrics,) == True

    # assert isinstance(modelConfigs, List) == True
    trainingMediator = training_mediator.ModelTrainingMediator(
        config=runConfig, modelConfigs=modelConfigs, run_id=trainRunId
    )
    assert trainingMediator is not None
    assert len(trainingMediator.MODEL_QUEUE) == 1
    currModelConfig = trainingMediator.MODEL_QUEUE[0]
    assert currModelConfig is not None
    assert currModelConfig.model_name == "ARIMA"
    assert currModelConfig.model_metric == "MAPE"
    assert bool(currModelConfig.metrics)


def testTrainMediatorData():
    trainRunId = 17
    runConfig = get_train_run_config(trainRunId)
    assert runConfig is not None
    modelConfigs = pred_db.get_model_configs_by_train_id(trainRunId)
    assert modelConfigs is not None
    # assert isinstance(modelConfigs, List) == True
    trainingMediator = training_mediator.ModelTrainingMediator(
        config=runConfig, modelConfigs=modelConfigs, run_id=trainRunId
    )
    assert trainingMediator is not None
    dataCommand = trainingMediator.dataCommand
    assert dataCommand is not None
    assert dataCommand.target_data_location is not None
    assert (
        dataCommand.target_data_location
        == "tests/resources/data/test_single_target.csv"
    )
    assert dataCommand.feature_data_location == ""
    assert "/score/" in trainingMediator.trainConfig.score_data_location


def testTrainingMediatorDataSplitArima():
    dataPath = S3_PRED_PATH + "h0500hn_ft-worth.csv"
    targetColumn = "H0500HN-FT-WORTH"
    trainingDict = {
        "training": {
            "train_name": "training-mediator",  # REQ
            "target_data_location": dataPath,
            "train_data_end_dtm": "2019-12-01",
            "test_data_end_dtm": "2020-06-01",
            "validation_data_end_dtm": "2020-11-01",
            "model_names": ["ARIMA"],
            "score_data_location": PREDICTION_CONTEXT.S3_TRAIN_RESULT_PATH,
            "prediction_steps": "0",
        }
    }
    config = pred_db.create_train_run_config(trainingDict, username="devin")
    assert config is not None
    assert config.train_run_id is not None
    assert config.train_run_id >= 0
    trainRunId = config.train_run_id
    modelConfigList = pred_db.get_model_configs_by_train_id(trainRunId)
    assert modelConfigList is not None
    assert len(modelConfigList) > 0

    trainingMediator = training_mediator.ModelTrainingMediator(
        config=config, modelConfigs=modelConfigList, run_id=trainRunId
    )
    assert trainingMediator is not None
    dataCommand = trainingMediator.dataCommand
    assert dataCommand is not None
    assert dataCommand.target_data_location is not None
    assert dataCommand.target_data_location == dataPath
    assert dataCommand.feature_data_location == ""
    assert (
        trainingMediator.trainConfig.score_data_location
        == PREDICTION_CONTEXT.S3_TRAIN_RESULT_PATH
    )
    assert trainingMediator.MODEL_QUEUE is not None
    assert len(trainingMediator.MODEL_QUEUE) > 0


def testTrainingMediatorDataSplitHolt(tempdir):
    dataPath = S3_PRED_PATH + "h0500hn_ft-worth.csv"
    targetColumn = "H0500HN-FT-WORTH"
    trainingDict = {
        "training": {
            "train_name": "training-mediator",  # REQ
            "target_data_location": dataPath,
            "train_data_end_dtm": "2019-12-01",
            "test_data_end_dtm": "2020-06-01",
            "validation_data_end_dtm": "2020-11-01",
            "model_names": ["HOLTWINTERS"],
            "score_data_location": f"{tempdir}/score",
            "prediction_steps": "0",
        }
    }
    config = pred_db.create_train_run_config(trainingDict, username="devin")
    assert config is not None
    assert config.train_run_id is not None
    assert config.train_run_id >= 0
    trainRunId = config.train_run_id
    modelConfigList = pred_db.get_model_configs_by_train_id(trainRunId)
    assert modelConfigList is not None
    assert len(modelConfigList) > 0

    trainingMediator = training_mediator.ModelTrainingMediator(
        config=config, modelConfigs=modelConfigList, run_id=trainRunId
    )
    assert trainingMediator is not None
    dataCommand = trainingMediator.dataCommand
    assert dataCommand is not None
    assert dataCommand.target_data_location is not None
    assert dataCommand.target_data_location == dataPath
    assert dataCommand.feature_data_location == ""
    assert (
        trainingMediator.trainConfig.score_data_location
        == trainingDict["training"]["score_data_location"]
    )
    assert trainingMediator.MODEL_QUEUE is not None
    assert len(trainingMediator.MODEL_QUEUE) > 0


@patch("ray.get", autospec=True)
@patch("ray.put", autospec=True)
def testDataPrep(ray_put, ray_get):
    data_store = {}
    counter = count()

    def ray_put_mock(data):
        id = next(counter)
        data_store[id] = data
        return id

    ray_put.side_effect = ray_put_mock
    ray_get.side_effect = lambda id: data_store[id]

    train_cmd = TrainingModel(
        train_model_id=1,
        train_name="ARIMA-test",
        train_description="Testing ARIMA",
        train_job_type="Sequential",
        model_location=PREDICTION_CONTEXT.S3_PRED_MODEL_PATH,
        model_name="ARIMA",
        model_config={},
        model_metric="MAPE",
        model_time_frequency="M",
        model_params={},
        model_hyperparams={},
        modeler=None,
        data_preprocessor=TSADataPreProcessor,
    )

    dataPath = S3_PRED_PATH + "h0500hn_ft-worth.csv"
    targetColumn = "H0500HN-FT-WORTH"
    trainRunId = 185
    runConfig = get_train_run_config(trainRunId)
    assert runConfig is not None
    assert runConfig.train_run_id == trainRunId
    modelConfigList = pred_db.get_model_configs_by_train_id(trainRunId)
    assert modelConfigList is not None
    # assert isinstance(modelConfigs, List) == True
    arimaConfig = modelConfigList[0]
    logger.info(f"ARIMA Config: {arimaConfig.model_config}")
    assert arimaConfig is not None
    logger.info(arimaConfig.model_config)
    assert arimaConfig.train_run_id.train_run_id == trainRunId

    trainingMediator = training_mediator.ModelTrainingMediator(
        config=runConfig, modelConfigs=modelConfigList, run_id=trainRunId
    )
    assert trainingMediator is not None

    dataCommand = trainingMediator.dataCommand
    assert dataCommand is not None
    assert dataCommand.target_data_location is not None
    assert dataCommand.target_data_location == dataPath
    assert dataCommand.feature_data_location == ""
    assert (
        trainingMediator.trainConfig.score_data_location
        == "tests/resources/data/train/training-mediator_2020-12-22-03.18.46.csv"
    )
    assert trainingMediator.MODEL_QUEUE is not None
    assert len(trainingMediator.MODEL_QUEUE) > 0

    trainingMediator._prepare_model_data(train_cmd)
    # The target data references object should be the 3rd thing stored after the
    assert trainingMediator.DATA_QUEUE_ID == 4

    # check that training, test, and validation data are stored in the data store and that they are subsets of each other in length
    assert data_store[0].shape == (120, 1)
    assert data_store[1].shape == (127, 1)
    assert data_store[2].shape == (131, 1)

    # Check that the target references are mapped correctly
    assert data_store[4] == {
        "H0500HN-FT-WORTH": {
            "data": {"train": 0, "test": 1, "valid": 2},
            "features": 3,
        }
    }


@pytest.mark.skip
def testTrainArima():
    trainRunId = 185
    runConfig = get_train_run_config(trainRunId)
    modelConfigList = pred_db.get_model_configs_by_train_id(trainRunId)
    trainingMediator = training_mediator.ModelTrainingMediator(
        config=runConfig, modelConfigs=modelConfigList, run_id=trainRunId
    )
    assert trainingMediator is not None

    result = trainingMediator.train_in_parallel()
    logger.info(f"Result of ARIMA training: {result}")
    assert len(trainingMediator.DATA_QUEUE_ID) == 1
    assert result is not None
    resultDf = pd.DataFrame.from_dict(result, orient="index")
    logger.info(f"ARIMA result as df: {resultDf}")

    # Validate Bias is calculated.
    assert resultDf is not None
    assert resultDf[2] is not None
    assert "bias" in resultDf[2]

    # s3_files.save_train_result(result, runConfig.score_data_location,
    # runConfig.train_name, datetime.now())


@pytest.mark.skip
def testTrainHolt():
    trainRunId = 186
    runConfig = get_train_run_config(trainRunId)
    modelConfigList = pred_db.get_model_configs_by_train_id(trainRunId)
    trainingMediator = training_mediator.ModelTrainingMediator(
        config=runConfig, modelConfigs=modelConfigList, run_id=trainRunId
    )
    assert trainingMediator is not None

    result = trainingMediator.train_in_parallel()
    logger.info(f"Result of HOLT training: {result}")
    assert len(trainingMediator.DATA_QUEUE_ID) == 1
    assert result is not None
    resultDf = pd.DataFrame.from_dict(result, orient="index")
    logger.info(f"HOLT result as df: {resultDf}")
    assert resultDf is not None


# ----------------------------TUNE------------------
def testTuneEnsembleMediatorCreate(ensemble_tune_config):
    runConfig = pred_db.create_train_run_config(ensemble_tune_config, username="devin")
    assert runConfig is not None
    trainRunId = runConfig.train_run_id

    modelConfigs = pred_db.get_model_configs_by_train_id(trainRunId)
    assert modelConfigs is not None
    assert len(modelConfigs) == len(PREDICTION_CONTEXT.ensemble_base_models)

    # assert isinstance(modelConfigs, List) == True
    trainingMediator = training_mediator.ModelTrainingMediator(
        config=runConfig, modelConfigs=modelConfigs, run_id=trainRunId
    )
    assert trainingMediator is not None
    assert len(trainingMediator.MODEL_QUEUE) == len(
        PREDICTION_CONTEXT.ensemble_base_models
    )
    for modelConfig in trainingMediator.MODEL_QUEUE:
        assert modelConfig is not None
        model_name = modelConfig.model_name
        assert len(model_name) > 0
        model_name = model_name.upper()
        assert model_name in PREDICTION_CONTEXT.ensemble_base_models
        config = modelConfig.model_config
        assert config is not None
        default_config = PREDICTION_CONTEXT.default_tune_config[model_name]
        assert config == default_config
        assert modelConfig.model_params
        assert isinstance(modelConfig.model_params, Dict) is True
        assert modelConfig.model_params == default_config["parameters"]
        assert modelConfig.model_hyperparams == default_config["hyperparameters"]


@pytest.mark.skip
def testTuneArima():
    trainRunId = 30
    runConfig = get_train_run_config(trainRunId)
    assert runConfig is not None
    assert runConfig.train_run_id == trainRunId
    modelConfigList = pred_db.get_model_configs_by_train_id(trainRunId)
    assert modelConfigList is not None
    # assert isinstance(modelConfigs, List) == True
    arimaConfig = modelConfigList[0]
    logger.info(f"ARIMA Config: {arimaConfig.model_config}")
    assert arimaConfig is not None
    logger.info(arimaConfig.model_config)
    assert arimaConfig.train_run_id.train_run_id == trainRunId
    trainingMediator = training_mediator.ModelTrainingMediator(
        config=runConfig, modelConfigs=modelConfigList, run_id=trainRunId
    )
    assert trainingMediator is not None
    trainingMediator.tune()
    time.sleep(2)
    assert len(trainingMediator.DATA_QUEUE_ID) > 0
    assert len(trainingMediator.TUNE_QUEUE) > 0


def testDispatchTasks():
    trainingDict = {
        "training": {
            "train_name": "model-mediator-test",  # REQ
            "target_data_location": f"{S3_PRED_PATH}/test_single_target.csv",
            "train_data_end_dtm": "2020-1-1",
            "test_data_end_dtm": "2020-6-1",
            "validation_data_end_dtm": "2020-12-1",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],  # REQ
            "prediction_steps": "0",
        }
    }
    config = pred_db.create_train_run_config(trainingDict, username="devin")


@pytest.mark.slow
def testTrainingMultipleFHR(tempdir):
    c3 = {
        "training": {
            "train_name": "model-minimum-config-jupyter",
            "target_data_location": f"{S3_PRED_PATH}/TrainingInputData_Transformed_Test.csv",
            "train_data_end_dtm": "2020-1-1",
            "test_data_end_dtm": "2020-6-1",
            "validation_data_end_dtm": "2020-12-1",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
            "prediction_steps": "0",
        }
    }
    config = pred_db.create_train_run_config(c3, username="test")
    assert config is not None
    assert config.train_run_id is not None
    assert config.train_run_id >= 0
    trainRunId = config.train_run_id
    modelConfigList = pred_db.get_model_configs_by_train_id(trainRunId)
    assert modelConfigList is not None
    assert len(modelConfigList) > 0

    trainingMediator = training_mediator.ModelTrainingMediator(
        config=config, modelConfigs=modelConfigList
    )
    assert trainingMediator is not None
    dataCommand = trainingMediator.dataCommand
    assert dataCommand is not None
    assert dataCommand.target_data_location is not None
    assert "TrainingInputData_Transformed_Test.csv" in dataCommand.target_data_location
    assert dataCommand.feature_data_location == ""
    assert (
        trainingMediator.trainConfig.score_data_location
        == PREDICTION_CONTEXT.S3_TRAIN_RESULT_PATH
    )
    assert trainingMediator.MODEL_QUEUE is not None
    assert len(trainingMediator.MODEL_QUEUE) > 0
    # Testing training
    result = trainingMediator.train_in_parallel()
    logger.info(f"Result of HOLT training: {result}")
    assert len(trainingMediator.DATA_QUEUE_ID) > 1
    assert result is not None
    resultDf = pd.DataFrame.from_dict(result, orient="index")
    logger.info(f"HOLT result as df: {resultDf}")
    assert resultDf is None


# -----------------------RAY TRAIN PARALLEL------------------
@pytest.mark.skip
def testTrainingParallel(tempdir):
    c3 = {
        "training": {
            "train_name": "model-minimum-config-jupyter",
            "target_data_location": f"{S3_PRED_PATH}/TrainingInputData_Transformed_Test.csv",  # TrainingInputData_Transformed_Test_Sample_2.csv",
            "train_data_end_dtm": "2020-1-1",
            "test_data_end_dtm": "2020-6-1",
            "validation_data_end_dtm": "2020-12-1",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
            "prediction_steps": "0",
        }
    }
    config = pred_db.create_train_run_config(c3, username="test")
    assert config is not None
    assert config.train_run_id is not None
    assert config.train_run_id >= 0
    trainRunId = config.train_run_id
    modelConfigList = pred_db.get_model_configs_by_train_id(trainRunId)
    assert modelConfigList is not None
    assert len(modelConfigList) > 0

    trainingMediator = training_mediator.ModelTrainingMediator(
        config=config, modelConfigs=modelConfigList
    )
    assert trainingMediator is not None
    dataCommand = trainingMediator.dataCommand
    assert dataCommand is not None
    assert dataCommand.target_data_location is not None
    # assert "TrainingInputData_Transformed_Test.csv" in dataCommand.target_data_location
    assert dataCommand.feature_data_location == ""
    assert (
        trainingMediator.trainConfig.score_data_location
        == PREDICTION_CONTEXT.S3_TRAIN_RESULT_PATH
    )
    assert trainingMediator.MODEL_QUEUE is not None
    assert len(trainingMediator.MODEL_QUEUE) > 0

    # Testing PARALLEL training
    # ray.init(ignore_reinit_error=True)
    ray.init(local_mode=True, num_cpus=mp.cpu_count())
    tic = time.perf_counter()
    result = trainingMediator.train_in_parallel()
    toc = time.perf_counter()
    logger.info(
        f"*********Result of training: {result}, time elapsed: {toc-tic:0.4f} seconds"
    )
    assert len(trainingMediator.DATA_QUEUE_ID) > 1
    assert result is None

    ray.shutdown()


# -----------------------RAY TUNE------------------
@pytest.mark.skip
def testTune():
    c3 = {
        "training": {
            "train_name": "model-minimum-config-jupyter",
            "train_task_type": "TUNING",
            "target_data_location": f"{S3_PRED_PATH}/TrainingInputData_Transformed_Test.csv",
            "train_data_end_dtm": "2020-1-1",
            "test_data_end_dtm": "2020-6-1",
            "validation_data_end_dtm": "2020-12-1",
            "model_names": ["ARIMA"],
        },
        "models": {
            "ARIMA": {
                "model_name": "ARIMA",
                "model_time_interval": "M",
                "hyperparam_alg": "GRID-SEARCH",
                "model_config": {
                    "parameters": {
                        "p": [1, 2, 3, 4, 5, 6, 7],
                        "d": [1, 2],
                        "q": [0, 1, 2],
                    },
                    "hyperparameters": {"disp": 0},
                },
            }
        },
    }
    config = pred_db.create_train_run_config(c3, username="test")
    assert config is not None
    assert config.train_run_id >= 0
    trainRunId = config.train_run_id
    modelConfigList = pred_db.get_model_configs_by_train_id(trainRunId)
    assert len(modelConfigList) > 0

    trainingMediator = training_mediator.ModelTrainingMediator(
        config=config, modelConfigs=modelConfigList, run_id=trainRunId
    )
    dataCommand = trainingMediator.dataCommand
    assert dataCommand.target_data_location is not None
    assert "TrainingInputData_Transformed_Test.csv" in dataCommand.target_data_location

    # Testing tuning
    logger.info("****Starting Ray for tune***")
    ray.init(local_mode=True, num_cpus=2)
    result = trainingMediator.tune()
    logger.info(f"*****Result of HOLT tuning: {type(result)}")
    dfAcc = result.dataframe(metric="mean_accuracy", mode="max")
    logger.info(f"# # #Accuracy df: {tabulate(dfAcc)}")
    # --------------------BEST RESULT--------------------
    best_trial = result.get_best_trial(metric="mean_accuracy", mode="max", scope="all")
    bestAcc = best_trial.metric_analysis["mean_accuracy"]["last"]
    best_config = result.get_best_config(
        metric="mean_accuracy", mode="max", scope="all"
    )
    logger.info(f"Best: acc = {bestAcc}, config = {best_config}")
    # logger.info(f"# # # #Best trial: {result.get_best_trial(metric='mean_accuracy', mode='max',scope='all')}")
    # logger.info(f"# # # #Best config: {result.get_best_config(metric='mean_accuracy', mode='max',scope='all')}")

    trialDf = result.trial_dataframes
    logger.info(f"Trial dataframes: {trialDf}")
    # logger.info(f"Trial df columns: {trialDf.columns}")
    df = result.dataframe()
    columns = ["model", "target", "params/p", "params/d", "params/q", "mean_accuracy"]
    # logger.info(f"Dataframe shape: {df.shape} and columns: {df.columns}")
    logger.info(
        f"Dataframes of trials: {tabulate(df[columns], headers='keys', tablefmt='psql')}"
    )
    dfDict = result.fetch_trial_dataframes()
    logger.info(f"trial_dfs: {dfDict}")
    for key, currDf in dfDict.items():
        logger.info(f"@@@@{key}:{currDf}")
    logger.info(f"Trials List: {result.trials}")
    logger.info(f"trial.logdir: {result.trials[0].logdir}")

    # trials = [trial for trial in result.trials if trial.status == Trial.TERMINATED]
    # logger.info(f"Trials: {trials}")

    # assert result is None
    assert result is not None

    ray.shutdown()
