import logging
from datetime import datetime

import pandas as pd
import pytest
from dateutil.relativedelta import *
from moto import mock_s3

from prediction.app.prediction_commands import DataCommand, TrainingModel
from prediction.app.worker.data_preprocessor.tsa import TSADataPreProcessor
from prediction.src.static import PREDICTION_CONTEXT

logger = logging.getLogger(PREDICTION_CONTEXT.logger_name)

S3_PRED_PATH = "tests/resources/data/"


@pytest.fixture
def training_cmd():
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
    return train_cmd


def testDataHandlerSingleTarget(training_cmd):
    dataHandler = TSADataPreProcessor()
    path = "tests/resources/data/test_single_target.csv"

    dataCommand = DataCommand(
        target_data_location=path,
        feature_data_location="",
        train_data_end_dtm=datetime.now(),
        test_data_end_dtm=datetime.now(),
        validation_data_end_dtm=datetime.now(),
        model_name="ARIMA",
    )
    print(f"{dataCommand=}")
    dataDict = dataHandler.handle(dataCommand, training_cmd)
    assert dataDict is not None
    assert len(dataDict) == 1
    assert "target1" in dataDict
    df = dataDict["target1"]["data"]
    assert df is not None
    assert isinstance(df, pd.DataFrame) == True


def testDataHandlerSingleFHR(training_cmd):
    dataHandler = TSADataPreProcessor()
    S3_DATA_PATH = S3_PRED_PATH + "h0500hn_ft-worth.csv"
    targetColumn = "H0500HN-FT-WORTH"
    dataCommand = DataCommand(
        target_data_location=S3_DATA_PATH,
        feature_data_location="",
        train_data_end_dtm=datetime.strptime("2019-12-01", "%Y-%m-%d"),
        test_data_end_dtm=datetime.strptime("2020-06-01", "%Y-%m-%d"),
        validation_data_end_dtm=datetime.strptime("2020-11-01", "%Y-%m-%d"),
        model_name="ARIMA",
        model_frequency="M",
    )
    dataDict = dataHandler.handle(dataCommand, training_cmd)
    assert dataDict is not None
    assert len(dataDict) == 1
    assert targetColumn in dataDict
    df = dataDict[targetColumn]["data"]
    logger.info(f"FHR Data: {df}")
    assert df is not None
    assert isinstance(df, pd.DataFrame) == True
    target, test, valid = dataHandler.split_train_test_valid(df=df, request=dataCommand)
    logger.info(f"Returned target data type: {type(target)}, cnt: {target.count()}")
    assert target is not None
    assert target.shape[0] > 0
    assert test is not None
    assert test.shape[0] > 0
    assert valid is not None
    assert valid.shape[0] > 0


def testDataHandlerMultipleFHR(training_cmd):
    dataHandler = TSADataPreProcessor()
    S3_DATA_PATH = f"{S3_PRED_PATH}TrainingInputData_Transformed_Test.csv"
    targetColumn = "19463:10184_LONGVIEW-FHR:13T55V"
    dataCommand = DataCommand(
        target_data_location=S3_DATA_PATH,
        feature_data_location="",
        train_data_end_dtm=datetime.strptime("2020-06-01", "%Y-%m-%d"),
        test_data_end_dtm=datetime.strptime("2020-09-01", "%Y-%m-%d"),
        validation_data_end_dtm=datetime.strptime("2020-12-01", "%Y-%m-%d"),
        model_name="ARIMA",
        model_frequency="M",
    )
    dataDict = dataHandler.handle(dataCommand, training_cmd)
    assert dataDict is not None
    assert len(dataDict) > 1
    assert targetColumn in dataDict
    df = dataDict[targetColumn]["data"]
    logger.info(f"FHR Data: {df}")
    assert df is not None
    assert isinstance(df, pd.DataFrame) == True
    target, test, valid = dataHandler.split_train_test_valid(df=df, request=dataCommand)
    logger.info(f"Returned target data type: {type(target)}, cnt: {target.count()}")
    assert target is not None
    assert target.shape[0] > 0
    assert test is not None
    assert test.shape[0] > 0
    assert valid is not None
    assert valid.shape[0] > 0


def testDataHandlerMultipleLeadingNulls(training_cmd):
    dataHandler = TSADataPreProcessor()
    S3_DATA_PATH = f"{S3_PRED_PATH}TrainingInputData_with_nulls.csv"
    targetColumn = "19463:10184_LONGVIEW-FHR:13T55V"
    dataCommand = DataCommand(
        target_data_location=S3_DATA_PATH,
        feature_data_location="",
        train_data_end_dtm=datetime.strptime("2020-06-01", "%Y-%m-%d"),
        test_data_end_dtm=datetime.strptime("2020-09-01", "%Y-%m-%d"),
        validation_data_end_dtm=datetime.strptime("2020-12-01", "%Y-%m-%d"),
        model_name="ARIMA",
        model_frequency="M",
    )
    dataDict = dataHandler.handle(dataCommand, training_cmd)
    assert dataDict is not None
    assert len(dataDict) > 1
    assert targetColumn in dataDict
    df = dataDict[targetColumn]["data"]
    logger.info(f"FHR Data: {df}")
    assert df is not None
    assert isinstance(df, pd.DataFrame) is True
    assert df.shape == (14, 1)
