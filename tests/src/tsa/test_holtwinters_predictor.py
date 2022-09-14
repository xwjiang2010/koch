import datetime
import json
import os
import sys
import time
import warnings
from pprint import pprint

import numpy as np
import pandas as pd
import pytest
import yaml

from prediction.src.tsa.holtwinters_predictor import HoltWintersPredictor

np.random.seed(0)

holt_p = {
    "holt_params": {"trend": "add", "damped": True, "seasonal": None},
    "holt_hyperparams": {
        "optimized": True,
        "use_boxcox": True,
        "remove_bias": True,
    },
}
dt_range = pd.date_range(start="07/1/2017", end="05/01/2020", freq="MS")  ## 36
n = len(dt_range)
values = np.random.uniform(0, 9000000, size=n)
df = pd.DataFrame({"y": values}, index=dt_range)
values_train = values[:30]
holt = HoltWintersPredictor(config=holt_p)
dt_single_target = pd.date_range(start="01/1/1990", end="02/05/1990", freq="D")  ## 36
target1 = np.array(
    [
        4,
        6,
        3,
        4,
        7,
        65,
        12,
        34,
        5,
        67,
        34,
        12,
        23,
        21,
        67,
        34,
        12,
        23,
        21,
        67,
        34,
        12,
        23,
        21,
        67,
        34,
        12,
        23,
        21,
        67,
        34,
        12,
        23,
        21,
        67,
        34,
    ],
    dtype=float,
)
df_target = pd.DataFrame({"y": target1}, index=dt_single_target)


def testNormalFit():
    holt_fit = holt.fit(dt_range, values)
    assert holt_fit.model is not None
    assert holt_fit.model_fit is not None


def testNormalPredict():
    pred_count = 2
    forecast = holt.predict(dt_range, values, prediction_count=pred_count)
    assert forecast is not None
    assert len(forecast) > 0
    assert len(forecast) == pred_count


def testNormalScore():
    pred_count = 2
    score = holt.score(dt_range, values, steps=pred_count)
    assert score is not None
    assert score > -100


def test_score_iterativel_normal():
    pred_count = 2
    score = holt.score_iteratively(dt_range, values, steps=pred_count)
    print(f"*********HOLT-WINTERS: {score}**********")
    assert score is not None
    assert score > -100


def test_normal_predict_target1():
    pred_count = 3
    holt_curr = HoltWintersPredictor(config=holt_p)
    forecast = holt_curr.predict(
        X=dt_single_target, y=target1, prediction_count=pred_count
    )
    assert forecast is not None
    assert len(forecast) > 0
    assert len(forecast) == pred_count


def test_normal_score_target1():
    pred_count = 3
    score = holt.score(dt_single_target, target1, steps=pred_count)
    assert score is not None
    assert score > -100


def test_score_iterativel_normal_target1():
    pred_count = 3
    score = holt.score_iteratively(dt_single_target, target1, steps=pred_count)
    # print(f"*********HOLT-WINTERS: {score}**********")
    assert score is not None
    assert score > -100


def test_calc_bias():
    pred_count = 2
    valid, y_hat = holt._get_prediction(dt_range, values, steps=pred_count)

    bias = holt.calc_bias(X=dt_range, y=values, steps=pred_count)
    expected_default_bias = np.sum(np.multiply((y_hat - valid) / valid, 1))

    assert bias != 0
    assert bias == expected_default_bias

    first_sample_all_weight = [1] + ([0] * (len(valid) - 1))

    bias_expected = (y_hat[0] - valid[0]) / valid[0]
    bias = holt.calc_bias(
        X=dt_range, y=values, steps=pred_count, sample_weights=first_sample_all_weight
    )

    assert bias == bias_expected
