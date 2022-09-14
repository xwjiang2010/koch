import logging
from datetime import datetime
from typing import Dict

import pytest
from pony.orm import *

import prediction.src.db.prediction_database as pred_db
from prediction.src.db.prediction_database import (
    TrainModelTrial,
    create_train_run_config,
    get_train_run_config,
)
from prediction.src.static import PREDICTION_CONTEXT

logger = logging.getLogger(PREDICTION_CONTEXT.logger_name)


@pytest.fixture
def train_run_config(tempdir):
    return {
        "master": {
            "exec_environment": "sequential",
            "data_storage_type": "s3",
            "data_file_type": "csv",
        },
        "training": {
            "train_name": "test-run-config",
            "train_description": "Testing insertion into the table 2",
            "train_task_type": "Model",
            "train_job_type": "Single-Target",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "feature_data_location": "",
            "score_data_location": f"{tempdir}/score/",
            "model_version": "abc",
            "data_version": "v1",
            "loss_function": "MAPE",
            "train_data_end_dtm": "2010-10-4",
            "test_data_end_dtm": "2011-01-01",
            "validation_data_end_dtm": "2012-01-01",
            "model_location": f"{tempdir}/model/",
            "time_interval": "M",
            "model_names": ["ARIMA"],
            "username": "allan",
            "prediction_steps": "0",
        },
    }


@pytest.fixture
def model_configs():
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


@pytest.fixture
def holt_config():
    return {
        "HOLTWINTERS": {
            "model_config": {
                "parameters": {"trend": "add", "damped": True, "seasonal": None},
                "hyperparams": {
                    "optimized": True,
                    "use_boxcox": True,
                    "remove_bias": True,
                },
            }
        }
    }


@pytest.fixture
def tsa_tune_config():
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
            "model_names": ["ARIMA", "HOLTWINTERS"],
            "prediction_steps": "0",
        },
    }


@pytest.fixture
def ensemble_tune_config():
    return {
        "master": {
            "exec_environment": "Parallel",
            "data_storage_type": "s3",
            "data_file_type": "csv",
            "resources": {
                "max_parallel_pods": 20,
            },
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


@pytest.fixture
def ensemble_tune_config_local(tempdir):
    return {
        "master": {
            "exec_environment": "Parallel",
            "data_storage_type": "s3",
            "data_file_type": "csv",
        },
        "training": {
            "train_task_type": "Tuning",
            "train_name": "KGSA Prediction Service Test",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": ["ENSEMBLE"],
            "prediction_steps": "0",
        },
    }


def testCreateTrainConfigMinimal():
    """trainingDict = {
        "training": {
            "train_name": "model-mediator-test",  ##REQ
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],  ##REQ
        }
    }"""
    trainingDict = {
        "training": {
            "train_name": "db-min-config-test",  ##REQ
            "target_data_location": "tests/resources/data/h0500hn_ft-worth.csv",
            "train_data_end_dtm": "2019-12-01",
            "test_data_end_dtm": "2020-06-01",
            "validation_data_end_dtm": "2020-11-01",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
            "prediction_steps": "0",
        }
    }
    ## Software version gets populated only during the runtime
    """softVer = PREDICTION_CONTEXT.kgsa_core_prediction_version
    logger.info(f"Software version: {softVer}")
    assert softVer is not None and softVer != "" """
    config = create_train_run_config(trainingDict, username="devin")
    assert config is not None
    assert config.train_run_id is not None
    assert config.train_run_id >= 0
    assert config.train_run_uuid is not None


def testCreateTrainConfig(train_run_config):
    config = create_train_run_config(train_run_config, username="devin")
    assert config is not None
    assert config.train_run_id is not None
    assert config.train_run_id >= 0
    assert config.train_run_uuid is not None
    masterParams = train_run_config["master"]
    assert config.train_exec_environment == masterParams["exec_environment"]


def testCreateTrainConfigHolt(train_run_config, holt_config):
    train_run_config["training"]["model_names"] = ["HOLTWINTERS"]
    train_run_config["models"] = holt_config
    logger.warn(f"HoltConfig: {train_run_config}")
    config = create_train_run_config(train_run_config, "devin")

    assert config is not None
    assert config.train_run_id is not None
    assert config.train_run_id >= 0
    assert config.train_run_uuid is not None
    masterParams = train_run_config["master"]
    assert config.train_exec_environment == masterParams["exec_environment"]


def testCreateTrainConfigMissingMaster(train_run_config, holt_config):
    del train_run_config["master"]
    train_run_config["training"]["model_names"] = ["HOLTWINTERS"]
    train_run_config["models"] = holt_config
    logger.warn(f"HoltConfig: {train_run_config}")
    config = create_train_run_config(train_run_config, "devin")

    assert config is not None
    assert config.train_run_id is not None
    assert config.train_run_id >= 0
    assert config.train_run_uuid is not None


def testCreateTrainModelConfigArima(model_configs, train_run_config):
    train_run_config["models"] = model_configs
    trainModelConfig = create_train_run_config(train_run_config, "devin")
    assert trainModelConfig is not None


def testCreateTrainModelConfigTune(train_run_config):
    hyperparam = {
        "ARIMA": {
            "model_name": "ARIMA",
            "model_time_interval": "M",
            "hyperparam_alg": "GRID-SEARCH",
            "model_config": {
                "parameters": {"p": [1, 7], "d": [1, 2], "q": [0, 2]},
                "hyperparameters": {"disp": 0},
            },
        }
    }
    train_run_config["models"] = hyperparam
    trainModelConfig = create_train_run_config(train_run_config, "devin")
    assert trainModelConfig is not None


def testCreateTuneModelConfigTSA(tsa_tune_config):
    config = create_train_run_config(tsa_tune_config, "devin")
    assert config is not None
    masterParams = tsa_tune_config["master"]
    assert config.train_exec_environment == masterParams["exec_environment"]
    config.train_run_id > 0
    model_configs = pred_db.get_model_configs_by_train_id(config.train_run_id)
    ##TrainRunModelConfig(train_run_id==runConfig)
    modelConfig = model_configs[0]
    assert modelConfig is not None
    model_name = modelConfig.model_name
    assert len(model_name) > 0
    model_name = model_name.upper()
    config = modelConfig.model_config
    assert config is not None
    default_config = PREDICTION_CONTEXT.default_tune_config[model_name]
    assert config == default_config
    ##assert isinstance(config, pony.orm.ormtypes.TrackedDict) == True
    arimaConfig = config["parameters"]
    assert arimaConfig is not None
    assert isinstance(arimaConfig, pony.orm.ormtypes.TrackedDict) == True
    assert isinstance(arimaConfig, Dict) == True
    assert arimaConfig == default_config["parameters"]


def testCreateTuneModelConfigEnsemble(ensemble_tune_config):
    config = create_train_run_config(ensemble_tune_config, "devin")
    assert config is not None
    assert (
        config.jobs_count
        == ensemble_tune_config["master"]["resources"]["max_parallel_pods"]
    )
    masterParams = ensemble_tune_config["master"]
    assert config.train_exec_environment == masterParams["exec_environment"]
    config.train_run_id > 0
    model_configs = pred_db.get_model_configs_by_train_id(config.train_run_id)
    assert len(model_configs) == len(PREDICTION_CONTEXT.ensemble_base_models)
    for modelConfig in model_configs:
        ##modelConfig = model_configs[0]
        assert modelConfig is not None
        model_name = modelConfig.model_name
        assert len(model_name) > 0
        model_name = model_name.upper()
        assert model_name in PREDICTION_CONTEXT.ensemble_base_models
        config = modelConfig.model_config
        assert config is not None
        default_config = PREDICTION_CONTEXT.default_tune_config[model_name]
        assert config == default_config
        ##assert isinstance(config, pony.orm.ormtypes.TrackedDict) == True
        arimaConfig = config["parameters"]
        assert arimaConfig is not None
        assert isinstance(arimaConfig, pony.orm.ormtypes.TrackedDict) == True
        assert isinstance(arimaConfig, Dict) == True
        assert arimaConfig == default_config["parameters"]


def testGetTrainRunConfig(seed_data):
    trainRunId = 5
    runConfig = get_train_run_config(trainRunId)
    ##runConfig = TrainRunConfig.get(train_run_id=trainRunId)
    assert runConfig is not None
    assert runConfig.train_run_id == trainRunId
    assert runConfig.train_run_uuid is not None
    """
    modelConfigSet = runConfig.models
    assert modelConfigSet is not None
    assert len(modelConfigSet) == 1
    """


def test_get_train_run_config_by_uuid(seed_data):
    trainRunId = 17
    config17 = get_train_run_config(trainRunId)
    runConfig = pred_db.get_train_run_config_by_uuid(
        config17.train_run_uuid, username="devin"
    )
    assert runConfig is not None
    assert runConfig.train_run_id == trainRunId
    assert runConfig.train_run_uuid is not None
    assert runConfig.train_data_end_dtm is not None
    assert runConfig.test_data_end_dtm is not None
    assert runConfig.validation_data_end_dtm is not None


def test_get_train_run_config_by_username():
    username = "allan"
    train_run_configs = pred_db.get_train_run_configs_by_username(username)
    assert train_run_configs is not None
    assert len(train_run_configs) > 0


def test_get_score_run_configs_by_username():
    username = "devin"
    train_run_configs = pred_db.get_score_run_configs_by_username(username)
    assert train_run_configs is not None
    assert len(train_run_configs) > 0


def testGetTrainRunModelConfigs(seed_data):
    trainRunId = 17
    model_configs = pred_db.get_model_configs_by_train_id(trainRunId)
    ##TrainRunModelConfig(train_run_id==runConfig)
    modelConfig = model_configs[0]
    assert modelConfig is not None
    config = modelConfig.model_config
    assert config is not None
    ##assert isinstance(config, pony.orm.ormtypes.TrackedDict) == True
    arimaConfig = config["parameters"]
    assert arimaConfig is not None
    assert isinstance(arimaConfig, pony.orm.ormtypes.TrackedDict) == True
    assert isinstance(arimaConfig, Dict) == True


def testUpdateTrainStatus(seed_data):
    trainRunId = 17
    pred_db.update_train_status(trainRunId=trainRunId, status="Running")
    runConfig = get_train_run_config(trainRunId)
    assert runConfig is not None
    assert runConfig.train_status == "Running"


def testGetTrainRunTrialsByTrainId(seed_data):
    trainRunId = 17
    trials = pred_db.get_train_run_trials_by_train_id(trainRunId)
    assert trials is not None
    assert len(trials) > 0


def testCreateTrainConfigEnsemble(seed_data, train_run_config):
    train_run_config["training"]["model_names"] = ["ENSEMBLE"]
    config = create_train_run_config(train_run_config, username="devin")

    assert config is not None
    assert config.model_type is not None
    assert config.model_type.upper() == "ENSEMBLE"

    assert config.train_run_id is not None
    assert config.train_run_id >= 0
    assert config.train_run_uuid is not None
    masterParams = train_run_config["master"]
    assert config.train_exec_environment == masterParams["exec_environment"]


##-------------------------------SCORING---------------------------------------------


@pytest.fixture
def score_uuid_config_dict(tempdir):
    return {
        "master": {
            "exec_environment": "sequential",
            "data_storage_type": "s3",
            "data_file_type": "csv",
        },
        "scoring": {
            "score_name": "test-ENSEMBLE",
            "score_description": "Testing insertion into the table 2",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "feature_data_location": "",
            "score_data_location": f"{tempdir}/score/",
            "data_version": "v1",
            "loss_function": "MAPE",
            "time_interval": "D",
            "model_names": ["ENSEMBLE"],
            "prediction_steps": 3,
            "prediction_count": 3,
            "username": "devin",
        },
    }


def test_create_score_train_uuid_ensemble(
    ensemble_tune_config_local, score_uuid_config_dict
):
    ##logger.info(f"{ensemble_tune_config_local}")
    config = create_train_run_config(ensemble_tune_config_local, username="devin")
    logger.debug(f"Created TrainRunConfig: {config}")
    trainRunId = config.train_run_id
    ## TrainRunModelConfig--------------------------------------
    trainRunUuid = config.train_run_uuid
    trainModelConfigs = pred_db.get_model_configs_by_train_id(config.train_run_id)
    assert len(trainModelConfigs) == len(PREDICTION_CONTEXT.ensemble_base_models)

    # Create train run trials------------------
    bestParams = {
        "PROPHET": {
            "growth": "linear",
            "holidays": "holidays",
            "changepoints": None,
            "interval_width": 0.8,
            "n_changepoints": 20,
            "seasonality_mode": "additive",
            "changepoint_range": 0.8,
            "daily_seasonality": "auto",
            "weekly_seasonality": "auto",
            "yearly_seasonality": "auto",
            "uncertainty_samples": 200,
            "holidays_prior_scale": 10.0,
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
        },
        "ARIMA": {"d": 0, "p": 1, "q": 3},
        "HOLTWINTERS": {
            "trend": "mul",
            "damped": False,
            "seasonal": "add",
            "seasonal_periods": 6,
        },
    }
    best_acc = 0.92341
    validation_acc = 0.8745
    for trainModelConfig in trainModelConfigs:
        trainModelId = trainModelConfig.train_run_model_id
        trial = TrainModelTrial(
            target_column="target1",
            model_loss_function="MAPE",
            model_config=bestParams[trainModelConfig.model_name],
            trial_status="Completed",
            train_score=best_acc,
            test_score=best_acc,
            validation_score=validation_acc,
            trial_start_ts=datetime.now(),
            trial_end_ts=datetime.now(),
            metrics={"bias": True},
        )
        createdTrial = pred_db.create_train_run_trial(trainModelId, trial)
        assert createdTrial
        assert createdTrial.metrics == {"bias": True}
        assert createdTrial.train_run_model_id.train_run_model_id == trainModelId

    trainTrials = pred_db.get_train_run_trials_by_train_id(trainRunId)
    assert len(trainTrials) == len(trainModelConfigs)

    for currTrial in trainTrials:
        assert currTrial.train_score == best_acc
        assert currTrial.validation_score == validation_acc

    ## ScoreRunConfig--------------------------------------
    logger.info(f"TrainRunConfig uuid: {trainRunUuid}")
    score_uuid_config_dict["scoring"]["train_run_id"] = trainRunUuid
    score_uuid_config_dict["username"] = "devin"
    scoreConfig = pred_db.create_score_run_config(score_uuid_config_dict)
    assert scoreConfig is not None
    logger.debug(f"ScoreConfig: {scoreConfig}")
    logger.info(f"id={scoreConfig.score_run_id},uuid={scoreConfig.score_run_uuid}")
    scoreId = scoreConfig.score_run_id
    assert scoreId >= 0
    assert scoreConfig.train_run_id == trainRunId

    ## Testing ScoreModelSummary----------------------------
    summaries = pred_db.get_score_model_summaries_by_score_id(scoreId)
    assert summaries is not None
    assert len(summaries) == len(trainTrials)
    summary = summaries[0]
    for summary in summaries:
        assert summary is not None
        assert summary.score_run_id.score_run_id == scoreId
        modelName = summary.model_name
        assert modelName in bestParams
        logger.debug(f"Summary config: {summary.model_config}")
        ##assert bestParams[modelName] in summary.model_config
    ##assert summary is None


@pytest.fixture
def scoreConfigDict(tempdir):
    return {
        "master": {
            "exec_environment": "sequential",
            "data_storage_type": "s3",
            "data_file_type": "csv",
        },
        "scoring": {
            "score_name": "test-arima",
            "score_description": "Testing insertion into the table 2",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "feature_data_location": "",
            "score_data_location": f"{tempdir}/score/",
            "data_version": "v1",
            "loss_function": "MAPE",
            "time_interval": "D",
            "model_names": ["ARIMA"],
            "prediction_steps": 3,
            "prediction_count": 3,
            "username": "devin",
        },
    }


@pytest.fixture
def scoreModelConfigsDict():
    return {
        "ARIMA": {
            "model_name": "ARIMA",
            "model_time_interval": "M",
            "model_config": {
                "parameters": {"p": 1, "d": 1, "q": 0},
                "hyperparameters": {"disp": 0},
            },
        },
        "scoring": {
            "username": "devin",
        },
    }


def testCreateScoreRunConfigMinimal():
    scoringDict = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["ARIMA"],
            "prediction_steps": 3,
            "username": "devin",
        }
    }

    scoreConfig = pred_db.create_score_run_config(scoringDict)
    assert scoreConfig is not None
    logger.info(f"ScoreConfig: {scoreConfig}")
    logger.info(f"id={scoreConfig.score_run_id},uuid={scoreConfig.score_run_uuid}")
    assert scoreConfig.score_run_id is not None
    ##assert modelsList is not None


def testCreateScoreRunConfig(scoreConfigDict, scoreModelConfigsDict):
    scoreConfigDict["models"] = scoreModelConfigsDict
    logger.info(f"Scoring input: {scoreConfigDict}")
    assert ("models" in scoreConfigDict) is True
    scoreConfig = pred_db.create_score_run_config(scoreConfigDict)
    assert scoreConfig is not None
    logger.info(f"ScoreConfig: {scoreConfig}")
    logger.info(f"id={scoreConfig.score_run_id},uuid={scoreConfig.score_run_uuid}")
    assert scoreConfig.score_run_id is not None
    ##assert modelsList is not None


def testCreateScoreRunConfigByTrainUUID(train_run_config, scoreConfigDict, holt_config):
    train_run_config["training"]["model_names"] = ["HOLTWINTERS"]
    train_run_config["models"] = holt_config
    logger.info(f"HoltConfig: {train_run_config}")
    config = create_train_run_config(train_run_config, username="devin")
    logger.info(f"Created TrainRunConfig: {config}")
    ## TrainRunModelConfig--------------------------------------
    trainRunUuid = config.train_run_uuid
    trainModelConfigs = pred_db.get_model_configs_by_train_id(config.train_run_id)
    assert len(trainModelConfigs) > 0
    ## ScoreRunConfig--------------------------------------
    logger.info(f"TrainRunConfig uuid: {trainRunUuid}")
    scoreConfigDict["scoring"]["train_run_id"] = trainRunUuid
    scoreConfig = pred_db.create_score_run_config(scoreConfigDict)
    assert scoreConfig is not None
    logger.info(f"ScoreConfig: {scoreConfig}")
    logger.info(f"id={scoreConfig.score_run_id},uuid={scoreConfig.score_run_uuid}")
    scoreId = scoreConfig.score_run_id
    assert scoreId >= 0
    ## Testing ScoreModelSummary----------------------------
    summaries = pred_db.get_score_model_summaries_by_score_id(scoreId)
    assert summaries is not None
    ##logger.info(f"Score Summaries: {summaries}")
    assert len(summaries) > 0
    summary = summaries[0]
    assert summary is not None
    assert summary.score_run_id.score_run_id == scoreId


def testGetScoreRunConfig(seed_data):
    scoreId = 1
    scoreConfig = pred_db.get_score_run_config(scoreId)
    assert scoreConfig is not None
    assert scoreConfig.score_run_id == scoreId
    assert scoreConfig.score_run_uuid is not None
    ## TODO assert scoreConfig.score_exec_environment == PREDICTION_CONTEXT.default_execution_environment
    assert (
        scoreConfig.score_data_storage_type
        == PREDICTION_CONTEXT.default_data_storage_type
    )
    assert scoreConfig.score_data_file_type == PREDICTION_CONTEXT.default_file_type


def testGetScoreRunConfigByUUID(seed_data):
    scoreUuid = "cf592d71-1d94-4bfb-969d-86c363fae5ad"
    scoreConfig = pred_db.get_score_run_config_by_uuid(scoreUuid, username="allan")
    assert scoreConfig is not None
    assert scoreConfig.score_run_id == 1
    assert scoreConfig.score_run_uuid is not None
    ## TODO assert scoreConfig.score_exec_environment == PREDICTION_CONTEXT.default_execution_environment
    assert (
        scoreConfig.score_data_storage_type
        == PREDICTION_CONTEXT.default_data_storage_type
    )
    assert scoreConfig.score_data_file_type == PREDICTION_CONTEXT.default_file_type


def testGetScoreSummariesByScoreId(seed_data):
    scoreId = 38
    summaries = pred_db.get_score_model_summaries_by_score_id(38)
    assert summaries is not None
    ##logger.info(f"Score Summaries: {summaries}")
    assert len(summaries) > 0
    summary = summaries[0]
    assert summary is not None
    assert summary.score_run_id.score_run_id == scoreId


def testCreateScoreRunConfigEnsmeble(tempdir):
    scoreConfigDict2 = {
        "master": {
            "exec_environment": "sequential",
            "data_storage_type": "s3",
            "data_file_type": "csv",
        },
        "scoring": {
            "score_name": "test-ENSEMBLE",
            "score_description": "Testing insertion into the table 2",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "feature_data_location": "",
            "score_data_location": f"{tempdir}/score/",
            "data_version": "v1",
            "loss_function": "MAPE",
            "time_interval": "D",
            "model_names": ["ENSEMBLE"],
            "prediction_steps": 3,
            "prediction_count": 3,
            "username": "devin",
        },
    }

    logger.info(f"Scoring input: {scoreConfigDict2}")

    scoreConfig = pred_db.create_score_run_config(scoreConfigDict2)
    assert scoreConfig is not None
    assert scoreConfig.model_type is not None
    assert scoreConfig.model_type.upper() == "ENSEMBLE"

    assert scoreConfig.score_run_id is not None
