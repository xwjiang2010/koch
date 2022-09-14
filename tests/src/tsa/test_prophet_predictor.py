import datetime
import json
import os
import sys
import time
from pprint import pprint

import numpy as np
import pandas as pd
import pytest
import yaml

from prediction.src.tsa.prophet_predictor import ProphetPredictor

np.random.seed(0)


prophet_p = {
    "prophet_params": {
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
    }
}
dt_range = pd.date_range(start="07/1/2017", end="05/01/2020", freq="MS")  ## 36
n = len(dt_range)
values = np.random.uniform(0, 9000000, size=n)
df = pd.DataFrame({"y": values}, index=dt_range)
values_train = values[:30]
features = pd.DataFrame(
    {
        "Date": pd.date_range(start="07/1/2017", end="08/01/2020", freq="MS"),
        "f1": np.random.uniform(200, 9000000, size=n + 3),
        "f2": np.random.uniform(2000, 9000000, size=n + 3),
        "f3": np.random.uniform(2000, 9000000, size=n + 3),
    }
)
features.set_index("Date", inplace=True)


prophet = ProphetPredictor(config=prophet_p)
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


# @pytest.mark.slow
def testNormalFit():
    prophet.fit(dt_range, values)
    assert prophet.model_fit is not None


# @pytest.mark.slow
def testNormalFitFeatures():
    prophet.fit(dt_range, values, features)
    assert prophet.model_fit is not None


##@pytest.mark.skip
def testNormalPredict():
    prophet.model_fit = False
    forecast = prophet.predict(dt_range, values)
    assert forecast is not None
    assert len(forecast) > 0


##@pytest.mark.skip
def testNormalPredictFeatures():
    prophet.model_fit = False
    forecast = prophet.predict(dt_range, values, features=features, prediction_count=3)
    assert forecast is not None
    assert len(forecast) > 0


##@pytest.mark.skip
def testNormalScore():
    prophet.model_fit = False
    score = prophet.score(dt_range, values)
    print(f"*****Prophet score: {score}")
    assert score is not None
    assert score > -100


##@pytest.mark.skip
def testNormalPredictSteps():
    pred_count = 2
    prophet.model_fit = False
    forecast = prophet.predict(dt_range, values, prediction_count=pred_count)
    assert forecast is not None
    ##print(f"*****Testing prediction count result: {forecast}")
    assert len(forecast) == pred_count


##@pytest.mark.skip
def testNormalPredictCountSteps():
    pred_count = 10
    steps = 12
    forecast = prophet.predict(
        dt_range, values, steps=steps, prediction_count=pred_count
    )
    assert forecast is not None
    ##print(f"Prophet reutrned type: {type(forecast)}")
    ##print(f"*****Testing prediction steps result: {forecast}")
    assert len(forecast) == pred_count


def test_score_iterativel_normal():
    pred_count = 2
    score = prophet.score_iteratively(dt_range, values, steps=pred_count)
    print(f"*********PROPHET: {score}**********")
    assert score is not None
    assert score > -100


def test_normal_predict_target1():
    pred_count = 3

    curr_prophet = ProphetPredictor(config=prophet_p)
    forecast = curr_prophet.predict(
        X=dt_single_target, y=target1, prediction_count=pred_count
    )
    assert forecast is not None
    assert len(forecast) > 0
    assert len(forecast) == pred_count


def test_normal_score_target1():
    pred_count = 3
    score = prophet.score(dt_single_target, target1, steps=pred_count)
    assert score is not None
    assert score != 0
    assert score > -100


def test_score_iterativel_normal_target1():
    pred_count = 3
    score = prophet.score_iteratively(dt_single_target, target1, steps=pred_count)
    # print(f"*********HOLT-WINTERS: {score}**********")
    assert score is not None
    assert score > -100


def test_calc_bias():
    pred_count = 2
    valid, y_hat = prophet._get_prediction(dt_range, values, steps=pred_count)

    bias = prophet.calc_bias(X=dt_range, y=values, steps=pred_count)
    expected_default_bias = np.sum(np.multiply((y_hat - valid) / valid, 1))

    assert bias != 0
    assert bias == expected_default_bias

    first_sample_all_weight = [1] + ([0] * (len(valid) - 1))

    bias_expected = (y_hat[0] - valid[0]) / valid[0]
    bias = prophet.calc_bias(
        X=dt_range, y=values, steps=pred_count, sample_weights=first_sample_all_weight
    )

    assert bias == bias_expected
