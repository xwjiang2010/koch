import datetime
import json
import logging
import os
import sys
import time

import boto3
import pandas as pd
import pytest
from moto import mock_s3

import prediction.src.s3_files as s3_files
from prediction.src.s3_files import read_s3_file
from prediction.src.static import PREDICTION_CONTEXT

logger = logging.getLogger(PREDICTION_CONTEXT.logger_name)


def testReadS3File():
    path = "tests/resources/data/test_multi_target.csv"
    df = read_s3_file(path)
    assert df is not None
    ## TODO
    assert isinstance(df, pd.DataFrame) == True


@pytest.mark.skip
def testCreateS3FileCSV():
    csvFile = s3_files.create_s3_path(
        s3Folder=PREDICTION_CONTEXT.S3_DATA_PATH, trainName="abc"
    )
    assert csvFile is not None
    assert len(csvFile) > 0
    assert csvFile.startswith(PREDICTION_CONTEXT.S3_DATA_PATH)
    assert csvFile.endswith("csv")


def testCreateS3FileJson():
    suffix = "json"
    csvFile = s3_files.create_s3_path(
        s3Folder=PREDICTION_CONTEXT.S3_DATA_PATH, trainName="abc", suffix=suffix
    )
    assert csvFile is not None
    assert len(csvFile) > 0
    assert csvFile.startswith(PREDICTION_CONTEXT.S3_DATA_PATH)
    assert csvFile.endswith(suffix)


@pytest.mark.skip
def testSaveTrainResultCSV():
    with tempfile.TemporaryDirectory() as tmpdirname:
        suffix = "csv"
        csvFile = s3_files.create_s3_path(
            s3Folder=tmpdirname, trainName="abc", suffix=suffix
        )
        assert csvFile is not None
        assert len(csvFile) > 0
        assert csvFile.startswith(tmpdirname)
        assert csvFile.endswith(suffix)

        logger.info(f"File name: {csvFile}")

        scoreConfigDict = {
            "exec_environment": "sequential",
            "data_storage_type": "s3",
            "data_file_type": "csv",
        }

        """     "scoring": {
                "train_name": "test-arima",
                "train_description": "Testing insertion into the table 2",
                "train_task_type": "Model",
                "train_job_type": "Single-Target",
                "target_data_location": "test_single_target.csv",
                "feature_data_location": "tests/tmp",
                "score_data_location": "score/",
                "model_version": "abc",
                "data_version": "v1",
                "loss_function": "MAPE",
                "train_data_end_dtm": "2010-10-4",
                "test_data_end_dtm": "2011-01-01",
                "validation_data_end_dtm": "2012-01-01",
                "model_location": "model/",
                "time_interval": "D",
                "model_names": ["ARIMA"],
                "prediction_steps": 3,
                "prediction_count": 2,
                }
            } """
        s3_files.save_train_result(
            results=scoreConfigDict, s3Path=csvFile, fileType=suffix
        )


@pytest.mark.skip
def testSaveTrainResultJson(tempdir):
    suffix = "json"
    csvFile = s3_files.create_s3_path(
        s3Folder=PREDICTION_CONTEXT.S3_DATA_PATH, trainName="abc", suffix=suffix
    )
    assert csvFile is not None
    assert len(csvFile) > 0
    assert csvFile.startswith(PREDICTION_CONTEXT.S3_DATA_PATH)
    assert csvFile.endswith(suffix)

    logger.info(f"File name: {csvFile}")

    scoreConfigDict = {
        "master": {
            "exec_environment": "sequential",
            "data_storage_type": "s3",
            "data_file_type": "csv",
        }
    }
    """     "scoring": {
            "train_name": "test-arima",
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
            "time_interval": "D",
            "model_names": ["ARIMA"],
            "prediction_steps": 3,
            "prediction_count": 2,
            }
        } """
    s3_files.save_train_result(results=scoreConfigDict, s3Path=csvFile, fileType=suffix)
