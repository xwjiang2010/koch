import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pytest
from dateutil.relativedelta import relativedelta

from prediction.src.tsa.arima_predictor import ARIMAPredictor

np.random.seed(0)

arima_p = {"arima_params": {"p": 1, "d": 1, "q": 0}}
arima = ARIMAPredictor(config=arima_p)

dt_range = pd.date_range(start="07/1/2017", end="05/01/2020", freq="MS")  ## 36
n = len(dt_range)
values = np.random.uniform(0, 9000000, size=n)
features = pd.DataFrame(
    {
        "Date": pd.date_range(start="07/1/2017", end="07/01/2020", freq="MS"),
        "f1": np.random.uniform(200, 9000000, size=n + 2),
        "f2": np.random.uniform(2000, 9000000, size=n + 2),
    }
)
features.set_index("Date", inplace=True)
df = pd.DataFrame({"y": values}, index=dt_range)
values_train = values[:30]
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
np.random.seed(0)
date_hrly_rng = pd.date_range(start="1/1/2018", end="10/01/2019", freq="MS")
y2 = np.random.randint(0, 9000000, size=(len(date_hrly_rng)))
df2 = pd.DataFrame({"y": y2}, index=date_hrly_rng)
df2["y"] = df["y"].astype(float)


def testFeatureFit():
    arima_fit = arima.fit(dt_range, values, features=features.copy())
    assert arima_fit.model_fit is not None
    model = arima_fit.model_fit
    summary = model.summary()
    assert summary is not None
    # Note that tables is a list. The table at index 1 is the "core" table.
    # Additionally, read_html puts dfs in a list, so we want index 0
    results_as_html = summary.tables[1].as_html()
    assert results_as_html is not None
    df1 = pd.read_html(results_as_html, header=0, index_col=0)[0]
    assert df1 is not None


def testNormalFit():
    arima_fit = arima.fit(dt_range, values)
    assert arima_fit.model_fit is not None
    model = arima_fit.model_fit
    summary = model.summary()
    assert summary is not None
    # Note that tables is a list. The table at index 1 is the "core" table.
    # Additionally, read_html puts dfs in a list, so we want index 0
    results_as_html = summary.tables[1].as_html()
    assert results_as_html is not None
    df1 = pd.read_html(results_as_html, header=0, index_col=0)[0]
    assert df1 is not None


def test_fit():
    freq = relativedelta(months=+1)
    train_data_end_dtm = datetime.datetime.fromisoformat("2019-04-01")
    target_train = df2[:train_data_end_dtm]

    x = (target_train.index,)
    y = target_train

    arima_fit = arima.fit(x, y)
    assert arima_fit.model_fit is not None
    model = arima_fit.model_fit
    summary = model.summary()
    assert summary is not None
    results_as_html = summary.tables[1].as_html()
    assert results_as_html is not None
    df1 = pd.read_html(results_as_html, header=0, index_col=0)[0]
    assert df1 is not None


def testNormalPredict():
    pred_count = 2
    forecast = arima.predict(dt_range, values, steps=pred_count)
    assert forecast is not None
    assert len(forecast) > 0
    ##print(f"*******ARIMA Forecast: {forecast}")
    ##print(f"ARIMA reutrned type: {type(forecast)}")
    assert len(forecast) == pred_count


def testFeaturePredict():
    pred_count = 2

    arima.model_fit = False  # make sure we
    forecast = arima.predict(dt_range, values, features=features, steps=pred_count)
    assert forecast is not None
    assert len(forecast) > 0
    ##print(f"*******ARIMA Forecast: {forecast}")
    ##print(f"ARIMA reutrned type: {type(forecast)}")
    assert len(forecast) == pred_count


def test_predict_validation_data():
    test_data_end_dtm = datetime.datetime.strptime(
        "2019-07-01", "%Y-%m-%d"
    )  # ,train_data_end_dtm + freq * 3
    df_test = df2[:test_data_end_dtm]
    test_x = (df_test.index,)
    test_y = df_test
    # test_x = df_test.reset_index().iloc[:, 0]
    # test_y = df_test.reset_index().iloc[:, 1]

    arima_fit = arima.fit(test_x, test_y)
    assert arima_fit.model_fit is not None

    valid_data_end_dtm = datetime.datetime.strptime(
        "2019-10-01", "%Y-%m-%d"
    )  # ,test_data_end_dtm + freq * 3
    df_valid = df2[:valid_data_end_dtm]
    valid_x = (df_valid.index,)
    valid_y = df_valid
    # valid_x = df_valid.reset_index().iloc[:, 0]
    # valid_y = df_valid.reset_index().iloc[:, 1]

    pred_count = 3
    forecast = arima.predict(valid_x, valid_y, steps=pred_count)
    assert forecast is not None
    assert len(forecast) > 0
    ##print(f"*******ARIMA Forecast: {forecast}")
    ##print(f"ARIMA reutrned type: {type(forecast)}")
    assert len(forecast) == pred_count


def testNormalScore():
    pred_count = 2
    score = arima.score(dt_range, values, steps=pred_count)
    assert score is not None
    assert score > -100


def test_score_iterativel_normal():
    pred_count = 2
    score = arima.score_iteratively(dt_range, values, steps=pred_count)
    # print(f"*********ARIMA: {score}**********")
    assert score is not None
    assert score > -100


def test_normal_predict_target1():
    pred_count = 3

    curr_arima = ARIMAPredictor(config=arima_p)
    forecast = curr_arima.predict(X=dt_single_target, y=target1, steps=pred_count)
    assert forecast is not None
    assert len(forecast) > 0
    assert len(forecast) == pred_count


def test_normal_score_target1():
    pred_count = 3
    score = arima.score(dt_single_target, target1, steps=pred_count)
    assert score is not None
    assert score != 0
    assert score != sys.maxsize
    assert score > -100


def test_score_iterativel_normal_target1():
    pred_count = 3
    score = arima.score_iteratively(dt_single_target, target1, steps=pred_count)
    # print(f"*********HOLT-WINTERS: {score}**********")
    assert score is not None
    assert score > -100


def test_score_validation_data():
    test_data_end_dtm = datetime.datetime.strptime(
        "2019-07-01", "%Y-%m-%d"
    )  # ,train_data_end_dtm + freq * 3
    df_test = df2[:test_data_end_dtm]
    test_x = (df_test.index,)
    test_y = df_test
    # test_x = df_test.reset_index().iloc[:, 0]
    # test_y = df_test.reset_index().iloc[:, 1]

    params = {"p": 1, "d": 1, "q": 0}
    arima_param = {"arima_params": params}
    arima_model = ARIMAPredictor(config=arima_param)
    arima_fit = arima_model.fit(X=test_x, y=test_y)
    assert arima_fit.model_fit is not None

    valid_data_end_dtm = datetime.datetime.strptime(
        "2019-10-01", "%Y-%m-%d"
    )  # ,test_data_end_dtm + freq * 3
    df_valid = df2[:valid_data_end_dtm]
    valid_x = (df_valid.index,)
    valid_y = df_valid
    # valid_x = df_valid.reset_index().iloc[:, 0]
    # valid_y = df_valid.reset_index().iloc[:, 1]
    assert type(valid_y) == pd.DataFrame

    pred_count = 3
    score = arima_model.score(X=valid_x, y=valid_y, steps=pred_count)
    assert score is not None
    assert score != 0
    assert score != sys.maxsize
    assert score >= -sys.maxsize
    assert score > 0


def test_score_iteratively_test_data():
    test_data_end_dtm = datetime.datetime.strptime(
        "2019-07-01", "%Y-%m-%d"
    )  # ,train_data_end_dtm + freq * 3
    df_test = df2[:test_data_end_dtm]
    test_x = (df_test.index,)
    test_y = df_test
    # test_x = df_test.reset_index().iloc[:, 0]
    # test_y = df_test.reset_index().iloc[:, 1]

    pred_count = 3
    params = {"p": 1, "d": 1, "q": 0}
    arima_param = {"arima_params": params}
    arima_model = ARIMAPredictor(config=arima_param)
    score = arima_model.score_iteratively(test_x, test_y, steps=pred_count)
    assert score is not None
    assert score != 0
    assert score != sys.maxsize
    assert score >= -sys.maxsize
    assert score > 0


def test_calc_bias():
    pred_count = 2
    valid, y_hat = arima._get_prediction(dt_range, values, steps=pred_count)

    bias = arima.calc_bias(X=dt_range, y=values, steps=pred_count)
    expected_default_bias = np.sum(np.multiply((y_hat - valid) / valid, 1))

    assert bias != 0
    assert bias == expected_default_bias

    first_sample_all_weight = [1] + ([0] * (len(valid) - 1))

    bias_expected = (y_hat[0] - valid[0]) / valid[0]
    bias = arima.calc_bias(
        X=dt_range, y=values, steps=pred_count, sample_weights=first_sample_all_weight
    )

    assert bias == bias_expected
