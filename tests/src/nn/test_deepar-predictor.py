import datetime
import json
import os
import sys
import time
import unittest
import warnings

import numpy as np
import pandas as pd
import pytest

from prediction.src.nn._deepar import DeepARPredictor
from prediction.src.utils import read_jsonlines

iam_role = "arn:aws:iam::254486207130:role/kbs-analytics-dev-fhr-demand-forecasting-SageMaker-ExecutionRoll"
deepar_p = {
    "num_samples": 1000,
    "pred_steps": 3,
    "FREQ": "MS",
    "train_instance_count": 1,
    "train_instance_type": "ml.m4.xlarge",
    "objective_metric_name": "test:mean_wQuantileLoss",
    "pred_instance_count": 1,
    "pred_instance_type": "ml.m4.xlarge",
    "max_jobs": 10,
    "max_parallel_jobs": 10,
    "iam_role": iam_role,
    "tags": [
        {"Key": "Name", "Value": "badrul.islam@kbslp.com"},
        {"Key": "costcenter", "Value": "54751"},
        {"Key": "blc", "Value": "1559"},
        {"Key": "itemid", "Value": "c3.ai"},
    ],
    "hyper_params": {
        "time_freq": "M",
        "early_stopping_patience": 10,
        "cardinality": "auto",
        "likelihood": "gaussian",
        "num_eval_samples": 1000,
        "test_quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "tunable_params": {
        "mini_batch_size": [10000, 10005],
        "epochs": [4000, 5002],
        "context_length": [3, 10],
        "num_cells": [1, 20],
        "num_layers": [1, 5],
        "dropout_rate": [0.0, 0.10],
        "embedding_dimension": [1, 15],
        "learning_rate": [1.0e-3, 1.0e-2],
    },
}
S3_FHR_PATH = "s3://badrul/predictions/FHR/"
S3_FHR_DATA_PATH = S3_FHR_PATH + "model_data/"
S3_FHR_MODEL_PATH = S3_FHR_PATH + "models/"
deepar_train_data = {
    "train": "s3://badrul/predictions/FHR/model_data/deepar/train/",
    "test": "s3://badrul/predictions/FHR/model_data/deepar/test/",
}
deepar_model_path = "s3://badrul/predictions/FHR/models/deepar"
curr_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
deepar_job_name = "deepar-FHRbadrul-" + curr_stamp


@pytest.fixture
def predict_json_data():
    file_path = "tests/resources/data/deepar_fhr_predict.json"
    features = read_jsonlines(file_path)
    return features


@pytest.mark.skip()
def testNormalFit():
    deepar = DeepARPredictor(config=deepar_p)
    assert deepar.job_name is None
    assert deepar.num_samples is not None
    deeparResults = deepar.fit(
        X=None,
        y=None,
        input_channels=deepar_train_data,
        job_name=deepar_job_name,
        output_path=deepar_model_path,
    )
    print(deeparResults)
    assert deeparResults
    assert len(deeparResults) > 0
    assert deepar.job_name is not None
    assert deepar.job_name == deepar_job_name
    ##TODO Check model path to make sure it exists


@pytest.mark.skip()
def testNormalPredict(predict_json_data):
    deepar = DeepARPredictor(config=deepar_p)
    forecast = deepar.predict(predict_json_data, pred_steps=3)
    assert forecast is not None
    assert len(forecastDates) > 0


@pytest.mark.skip()
def test_pred_on_tuned_model(predict_json_data):
    deepar = DeepARPredictor(
        config=deepar_p, tuned_job_name="deepar-FHRbadrul-20220210-2140"
    )
    forecast = deepar.predict(X=predict_json_data, pred_steps=3)
    assert forecast is not None
    assert len(forecast) > 0
