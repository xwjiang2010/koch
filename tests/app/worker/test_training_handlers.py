import logging
import sys
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

import prediction.app.prediction_commands as pred_commands
from prediction.app.prediction_commands import TrainingModel
from prediction.app.worker.data_preprocessor.tsa import TSADataPreProcessor
from prediction.app.worker.training_handlers import (
    DeepARModeler,
    DeepLearningModeler,
    TSAModeler,
)
from prediction.src.static import PREDICTION_CONTEXT

S3_PRED_PATH = "tests/resources/data/"

logger = logging.getLogger(PREDICTION_CONTEXT.logger_name)


np.random.seed(PREDICTION_CONTEXT.RANDOM_SEED)
date_hrly_rng = pd.date_range(start="1/1/2018", end="10/01/2019", freq="MS")
n = len(date_hrly_rng)
y = np.random.randint(0, 9000000, size=n)
df = pd.DataFrame({"y": y}, index=date_hrly_rng)
df["y"] = df["y"].astype(float)

features = pd.DataFrame(
    {
        "Date": date_hrly_rng,
        "f1": np.random.uniform(200, 9000000, size=n),
        "f2": np.random.uniform(2000, 9000000, size=n),
    }
)
features.set_index("Date", inplace=True)

logger.info(f"Data: {df}")
logger.info(f"Features: {features}")


@patch("ray.get", autospec=True)
def testArimaModeler(ray_get):
    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}, "features": 4},
        },
        1: df,
        2: None,
        3: None,
        4: None,
    }
    ray_get.side_effect = lambda ref: object_store[ref]

    arima = TSAModeler(target_references_id=0)
    assert arima is not None

    trainCmd = TrainingModel(
        train_model_id=1,
        train_name="ARIMA-test",
        train_description="Testing ARIMA",
        train_job_type="Sequential",
        model_location="s3://",
        model_name="ARIMA",
        model_config=None,
        model_metric="MAPE",
        model_time_frequency="M",
        model_params={"p": 1, "d": 1, "q": 0},
        model_hyperparams={},
        modeler=None,
        data_preprocessor=None,
    )

    error, validation_error, metrics = arima.handle(trainCmd, target="target_1")
    assert error is not None
    assert error >= 0
    assert validation_error is None
    assert metrics is not None


@patch("ray.get", autospec=True)
def testArimaModelerTrainTestvalid(ray_get):
    freq = relativedelta(months=+1)
    train_data_end_dtm = datetime.fromisoformat("2019-04-01")

    test_data_end_dtm = train_data_end_dtm + freq * 3
    valid_data_end_dtm = test_data_end_dtm + freq * 3
    target_train = df[:train_data_end_dtm]
    target_test = df[:test_data_end_dtm]
    target_valid = df[:valid_data_end_dtm]
    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}},
        },
        1: target_train,
        2: target_test,
        3: target_valid,
    }
    ray_get.side_effect = lambda ref: object_store[ref]

    arima = TSAModeler(
        target_references_id=0,
    )
    assert arima is not None
    assert arima.target_references_id == 0

    trainCmd = TrainingModel(
        train_model_id=1,
        train_name="ARIMA-test",
        train_description="Testing ARIMA",
        train_job_type="Sequential",
        model_location="s3://",
        model_name="ARIMA",
        model_config=None,
        model_metric="MAPE",
        model_time_frequency="M",
        model_params={"p": 1, "d": 1, "q": 0},
        model_hyperparams={},
        modeler=None,
        data_preprocessor=None,
        metrics={},
    )

    error, validation_error, metrics = arima.handle(trainCmd, target="target_1")
    assert error is not None
    assert error >= 0
    assert validation_error is not None
    assert validation_error >= 0
    assert error != sys.maxsize
    assert validation_error != sys.maxsize
    assert metrics == {}


@patch("ray.get", autospec=True)
def testArimaModelerTrainTestvalidWithFeatures(ray_get):
    freq = relativedelta(months=+1)
    train_data_end_dtm = datetime.fromisoformat("2019-04-01")

    test_data_end_dtm = train_data_end_dtm + freq * 3
    valid_data_end_dtm = test_data_end_dtm + freq * 3
    target_train = df[:train_data_end_dtm]
    target_test = df[:test_data_end_dtm]
    target_valid = df[:valid_data_end_dtm]

    feature_predict = pd.DataFrame(
        [
            np.random.uniform(200, 9000000, size=2),
            np.random.uniform(200, 9000000, size=2),
            np.random.uniform(200, 9000000, size=2),
        ],
        columns=list(["f1", "f2"]),
        index=pd.date_range(start="11/1/2019", end="01/01/2020", freq="MS"),
    )

    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}, "features": 4},
        },
        1: target_train,
        2: target_test,
        3: target_valid,
        4: features.append(feature_predict),
    }
    ray_get.side_effect = lambda ref: object_store[ref]

    arima = TSAModeler(target_references_id=0)
    assert arima is not None
    trainCmd = TrainingModel(
        train_name="ARIMA-test",
        train_description="Testing ARIMA",
        train_job_type="Sequential",
        model_location="s3://",
        model_name="ARIMA",
        model_config=None,
        model_metric="MAPE",
        model_time_frequency="M",
        model_params={"p": 1, "d": 1, "q": 0},
        model_hyperparams={},
        train_model_id=1,
        modeler=None,
        data_preprocessor=None,
        metrics={"bias": True},
    )

    error, validation_error, metrics = arima.handle(trainCmd, target="target_1")
    assert error is not None
    assert error >= 0
    assert validation_error is not None
    assert validation_error >= 0
    assert error != sys.maxsize
    assert validation_error != sys.maxsize
    assert metrics["bias"] is not None


@patch("ray.get", autospec=True)
def testHoltwinter(ray_get):
    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}, "features": 4},
        },
        1: df,
        2: None,
        3: None,
        4: None,
    }
    ray_get.side_effect = lambda ref: object_store[ref]
    holt = TSAModeler(target_references_id=0)
    assert holt is not None
    trainCmd = TrainingModel(
        train_model_id=1,
        train_name="Holwinter-test",
        train_description="Testing Holtwinters",
        train_job_type="Sequential",
        model_location="s3://",
        model_name="HOLTWINTERS",
        model_config=None,
        model_metric="MAPE",
        model_time_frequency="M",
        model_params={"trend": "add", "damped": True, "seasonal": None},
        model_hyperparams={"optimized": True, "use_boxcox": True, "remove_bias": True},
        modeler=None,
        data_preprocessor=None,
        metrics={},
    )
    error, validation_error, metrics = holt.handle(trainCmd, target="target_1")
    assert error is not None
    ## TODO assert error >= 0


@patch("ray.get", autospec=True)
@patch("ray.put", autospec=True)
def HoltWinterFHR(ray_put, ray_get):
    dataHandler = pred_handlers.DataHandler()
    S3_DATA_PATH = S3_PRED_PATH + "h0500hn_ft-worth.csv"
    targetColumn = "H0500HN-FT-WORTH"
    dataCommand = pred_commands.DataCommand(
        target_data_location=S3_DATA_PATH,
        feature_data_location="",
        train_data_end_dtm=datetime.strptime("2019-12-01", "%Y-%m-%d"),
        test_data_end_dtm=datetime.strptime("2020-06-01", "%Y-%m-%d"),
        validation_data_end_dtm=datetime.strptime("2020-11-01", "%Y-%m-%d"),
        model_name="ARIMA",
        model_frequency="M",
    )
    dataDict = dataHandler.handle(dataCommand)
    assert dataDict is not None
    assert len(dataDict) == 1
    assert targetColumn in dataDict
    dfFHR = dataDict[targetColumn]["data"]
    logger.info(f"FHR Data: {dfFHR}")
    assert dfFHR is not None
    assert isinstance(dfFHR, pd.DataFrame) == True
    target, test, valid = dataHandler.split_train_test_valid(
        df=dfFHR, request=dataCommand
    )
    logger.info(f"Returned target data type: {type(target)}, cnt: {target.count()}")
    assert target is not None
    assert target.shape[0] > 0
    assert test is not None
    assert test.shape[0] > 0
    assert valid is not None
    assert valid.shape[0] > 0
    holtConfig = PREDICTION_CONTEXT.default_models_config["HOLTWINTERS"]
    trainCmd = TrainingModel(
        train_model_id=1,
        train_name="Holwinter-test-FHR",
        train_description="Testing Holtwinters train,test and validation",
        train_job_type="Sequential",
        model_location=PREDICTION_CONTEXT.S3_PRED_MODEL_PATH,
        model_name="HOLTWINTERS",
        model_config=holtConfig,
        model_metric="MAPE",
        model_time_frequency="M",
        model_params=holtConfig["parameters"],
        ##{"trend": "add", "damped": True, "seasonal": None},
        model_hyperparams=holtConfig["hyperparameters"],
        ##{"optimized": True, "use_boxcox": True, "remove_bias": True},
        modeler=None,
        data_preprocessor=None,
        metrics={},
    )
    logger.info(f"Holtwinter target data: {target}")
    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}, "features": 4},
        },
        1: target,
        2: test,
        3: valid,
        4: None,
    }
    ray_get.side_effect = lambda ref: object_store[ref]
    holt = TSAModeler(
        target_references_id=0,
    )
    assert holt is not None

    trainError, validation_error, metrics = holt.handle(trainCmd, target="target_1")
    assert trainError is not None
    assert trainError >= 0
    assert validation_error is not None
    assert validation_error < sys.maxsize
    assert validation_error >= 0

    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}, "features": 4},
        },
        1: test,
        2: test,
        3: None,
        4: None,
    }
    ray_get.side_effect = lambda ref: object_store[ref]
    testError, validation_error, metrics = holt.handle(trainCmd, target="target_1")
    assert testError is not None
    assert testError < sys.maxsize
    assert testError >= 0
    assert validation_error is None

    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}, "features": 4},
        },
        1: valid,
        2: valid,
        3: None,
        4: None,
    }
    ray_get.side_effect = lambda ref: object_store[ref]
    testValid, validation_error, metrics = holt.handle(trainCmd, target="target_1")
    assert testValid is not None
    assert testValid >= 0
    assert testValid < sys.maxsize
    assert validation_error is None
    assert metrics == {}


@pytest.mark.skip
@patch("ray.get", autospec=True)
@patch("ray.put", autospec=True)
def testProphetFHR(ray_put, ray_get):
    data_preprocessor = TSADataPreProcessor()
    S3_PRED_PATH = "tests/resources/data/"
    S3_DATA_PATH = S3_PRED_PATH + "h0500hn_ft-worth.csv"
    targetColumn = "H0500HN-FT-WORTH"
    dataCommand = pred_commands.DataCommand(
        target_data_location=S3_DATA_PATH,
        feature_data_location="",
        train_data_end_dtm=datetime.strptime("2019-12-01", "%Y-%m-%d"),
        test_data_end_dtm=datetime.strptime("2020-06-01", "%Y-%m-%d"),
        validation_data_end_dtm=datetime.strptime("2020-11-01", "%Y-%m-%d"),
        model_name="ARIMA",
        model_frequency="M",
    )
    dataDict = data_preprocessor.handle(dataCommand)
    assert dataDict is not None
    assert len(dataDict) == 1
    assert targetColumn in dataDict
    dfFHR = dataDict[targetColumn]["data"]
    logger.info(f"FHR Data: {dfFHR}")
    assert dfFHR is not None
    assert isinstance(dfFHR, pd.DataFrame) == True
    target, test, valid = data_preprocessor.split_train_test_valid(
        df=dfFHR, request=dataCommand
    )

    logger.info(f"Returned target data type: {type(target)}, cnt: {target.count()}")
    assert target is not None
    assert target.shape[0] > 0
    assert test is not None
    assert test.shape[0] > 0
    assert valid is not None
    assert valid.shape[0] > 0
    prophetConfig = PREDICTION_CONTEXT.default_models_config["PROPHET"]
    trainCmd = TrainingModel(
        train_model_id=1,
        train_name="Prophet-test-FHR",
        train_description="Testing Holtwinters train,test and validation",
        train_job_type="Sequential",
        model_location=PREDICTION_CONTEXT.S3_PRED_MODEL_PATH,
        model_name="PROPHET",
        model_config=prophetConfig,
        model_metric="MAPE",
        model_time_frequency="M",
        model_params=prophetConfig["parameters"],
        model_hyperparams=prophetConfig["hyperparameters"],
        modeler=None,
        data_preprocessor=None,
    )

    logger.info(f"Prophet target data: {target}")
    prophet = TSAModeler(target_references_id=0)
    assert prophet is not None

    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}, "features": 4},
        },
        1: target,
        2: target,
        3: None,
        4: None,
    }
    ray_get.side_effect = lambda ref: object_store[ref]
    trainError, validation_error, metrics = prophet.handle(trainCmd, target="target_1")
    assert trainError is not None
    assert trainError >= 0

    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}, "features": 4},
        },
        1: test,
        2: test,
        3: None,
        4: None,
    }
    ray_get.side_effect = lambda ref: object_store[ref]
    prophet = TSAModeler(target_references_id=0)
    testError, validation_error, metrics = prophet.handle(trainCmd, target="target_1")
    assert testError is not None
    assert testError >= 0

    object_store = {
        0: {"target_1": {"data": {"train": 1, "test": 2, "valid": 3}}},
        1: valid,
        2: valid,
        3: None,
    }
    ray_get.side_effect = lambda ref: object_store[ref]
    prophet = TSAModeler(target_references_id=0)
    testValid, validation_error, metrics = prophet.handle(trainCmd, target="target_1")
    assert testValid is not None
    assert testValid >= 0


# @pytest.mark.skip
@patch("ray.get", autospec=True)
@patch("ray.put", autospec=True)
def testProphet(ray_put, ray_get):
    prophet = TSAModeler(target_references_id=0)
    assert prophet is not None
    trainCmd = TrainingModel(
        train_model_id=1,
        train_name="Prophet-test",
        train_description="Testing Prophet",
        train_job_type="Sequential",
        model_location="s3://",
        model_name="PROPHET",
        model_config=None,
        model_metric="MAPE",
        model_time_frequency="M",
        model_params={
            "growth": "linear",
            "changepoints": None,
            "n_changepoints": 25,
            "changepoint_range": 0.8,
            "yearly_seasonality": "auto",
            "weekly_seasonality": "auto",
            "daily_seasonality": "auto",
            "holidays": None,
            "seasonality_mode": "additive",
            "seasonality_prior_scale": 10.0,
            "holidays_prior_scale": 10.0,
            "changepoint_prior_scale": 0.05,
            "interval_width": 0.8,
            "uncertainty_samples": 1000,
        },
        model_hyperparams={},
        modeler=None,
        data_preprocessor=None,
    )
    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}},
        },
        1: df,
        2: df,
        3: None,
    }
    ray_get.side_effect = lambda ref: object_store[ref]
    error, validation_error, metrics = prophet.handle(trainCmd, target="target_1")
    assert error is not None
    assert error >= 0
    assert validation_error is None


@pytest.mark.skip
def test_deepar():
    """ Test DeepAR Model """
    deepar_modeler = DeepARModeler(trainingData=df, testData=None)
    assert deepar_modeler.data is not None
    assert deepar_modeler.test_data is None
    train_command = TrainingModel(
        train_model_id=1,
        train_name="deepar-test",
        train_description="Test DeepAR",
        train_job_type="Sequential",
        model_location="s3://",
        model_name="DEEPAR",
        model_config=None,
        model_metric="MAPE",
        model_time_frequency="M",
        model_params={
            "num_samples": 1000,
            "pred_steps": 3,
            "FREQ": "MS",
            "train_instance_count": 1,
            "train_instance_type": "ml.m4.xlarge",
            "objective_metric_name": "test:mean_wQuantileLoss",
            "pred_instance_count": 1,
            "pred_instance_type": "ml.m4.xlarge",
            "max_jobs": 10,
            "max_parallel_jobs": 1,
            "iam_role": "arn:aws:iam::254486207130:role/kbs-analytics-dev-fhr-demand-forecasting-SageMaker-ExecutionRoll",
            "tags": [
                {"Key": "Name", "Value": "kgsalasercats@kochind.onmicrosoft.com"},
                {"Key": "costcenter", "Value": "54751"},
                {"Key": "blc", "Value": "1559"},
            ],
            "mini_batch_size": [10000, 10005],
            "epochs": [4000, 5002],
            "context_length": [3, 10],
            "num_cells": [1, 20],
            "num_layers": [1, 5],
            "dropout_rate": [0.0, 0.10],
            "embedding_dimension": [1, 15],
            "learning_rate": [1.0e-3, 1.0e-2],
        },
        model_hyperparams={
            "time_freq": "M",
            "early_stopping_patience": 10,
            "cardinality": "auto",
            "likelihood": "gaussian",
            "num_eval_samples": 1000,
            "test_quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    )
    error = deepar_modeler.handle(train_command)
    assert error is not None
    assert error >= 0
