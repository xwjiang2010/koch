import datetime
import json
import logging
import os
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from prediction.app.prediction_commands import DataCommand, ScoringCommand
from prediction.app.worker.training_handlers import (
    DeepARModeler,
    DeepLearningModeler,
    TSAModeler,
)
from prediction.src.static import PREDICTION_CONTEXT

logger = logging.getLogger(PREDICTION_CONTEXT.logger_name)

date_hrly_rng = pd.date_range(start="1/1/2018", end="08/01/2018", freq="MS")
y = np.random.randint(0, 9000000, size=(len(date_hrly_rng)))
df = pd.DataFrame({"y": y}, index=date_hrly_rng)

logger.info(f"Data: {df}")


@patch("ray.get", autospec=True)
def testArimaScorer(ray_get):
    arima = TSAModeler(target_references_id=0)
    ##TSAScorer(trainingData=df, testData=None)
    assert arima is not None
    scoreModelConfigs = {
        "parameters": {"p": 1, "d": 1, "q": 0},
        "hyperparameters": {"disp": 0},
    }

    scoreCmd = ScoringCommand(
        score_name="Testing score command",
        target_column="target1",
        model_name="ARIMA",
        model_config=scoreModelConfigs,
        score_status="Created",
        prediction_steps=3,
        score_time_frequency="D",
    )
    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}, "features": 4},
        },
        1: df,
        2: df,
        3: None,
        4: None,
    }
    ray_get.side_effect = lambda ref: object_store[ref]

    forecast = arima.handle(scoreCmd, target="target_1")
    ##logger.info(f"ARIMA Forecast: {forecast}")
    assert forecast is not None
    assert len(forecast) == scoreCmd.prediction_count


@patch("ray.get", autospec=True)
def testHoltwinterScorer(ray_get):
    holt = TSAModeler(target_references_id=0)
    assert holt is not None
    scoreModelConfigs = {
        "parameters": {"trend": "add", "damped": True, "seasonal": None},
        "hyperparameters": {"optimized": True, "use_boxcox": True, "remove_bias": True},
    }

    scoreCmd = ScoringCommand(
        score_name="Testing score command",
        target_column="target1",
        model_name="HOLTWINTERS",
        model_config=scoreModelConfigs,
        score_status="Created",
        prediction_steps=3,
        score_time_frequency="D",
    )
    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}, "features": 4},
        },
        1: df,
        2: df,
        3: None,
        4: None,
    }
    ray_get.side_effect = lambda ref: object_store[ref]
    forecast = holt.handle(scoreCmd, target="target_1")
    ##logger.info(f"ARIMA Forecast: {forecast}")
    assert forecast is not None
    assert len(forecast) == scoreCmd.prediction_count


@patch("ray.get", autospec=True)
def testProphetScorer(ray_get):
    prophet = TSAModeler(target_references_id=0)
    assert prophet is not None
    scoreModelConfigs = {
        "parameters": {
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
        "hyperparameters": {},
    }

    scoreCmd = ScoringCommand(
        score_name="Testing score command",
        target_column="target1",
        model_name="PROPHET",
        model_config=scoreModelConfigs,
        score_status="Created",
        prediction_steps=3,
        prediction_count=3,
        score_time_frequency="D",
    )
    object_store = {
        0: {
            "target_1": {"data": {"train": 1, "test": 2, "valid": 3}, "features": 4},
        },
        1: df,
        2: df,
        3: None,
        4: None,
    }
    ray_get.side_effect = lambda ref: object_store[ref]
    forecast = prophet.handle(scoreCmd, target="target_1")
    ##logger.info(f"ARIMA Forecast: {forecast}")
    assert forecast is not None
    assert len(forecast) > 0


# TODO, wae, figure out how to expand this information below into the test_lstm_scorer method

## Use the below tunable params if using Sagemaker Hyperparameter tuning job
lstm_p = {
    "pred_steps": 3,
    "FREQ": "MS",
    "train_instance_count": 1,
    "train_instance_type": "ml.m4.10xlarge",  # ml.c5.4xlarge
    "job_name_prefix": "",  # update
    "early_stopping_type": "Auto",
    "max_jobs": 30,
    "max_parallel_jobs": 1,
    "tunable_params": {
        "epochs": 50,
        "context-length": 4,
        "lr": [0.001, 0.01],
        "hidden-dim": 10,
        "num-layers": 1,
        "bias": True,
        "fully-connected": True,
    },
    "hyper_params": {
        "batch_size": 10000,
        "test-batch-size": 10000,
        "log-interval": 1000,
        "dropout-eval": 0.2,
        "normalized": True,
        "test-loss-func": "mape",
        "loss-func": "mape",
    },
}


@pytest.mark.skip
def test_deepar_score():
    """ Test DeepAR scoring flow """
    deep_ar_modeler = DeepARModeler(trainingData=df, testData=None)
    assert deep_ar_modeler.data is not None
    assert deep_ar_modeler.test_data is None

    # TODO: More assertions?

    score_model_configs = {
        "parameters": {
            "mini_batch_size": [10000, 10005],
            "epochs": [4000, 5002],
            "context_length": [3, 10],
            "num_cells": [1, 20],
            "num_layers": [1, 5],
            "dropout_rate": [0.0, 0.10],
            "embedding_dimension": [1, 15],
            "learning_rate": [1.0e-3, 1.0e-2],
        },
        "hyperparameters": {
            "time_freq": "M",
            "early_stopping_patience": 10,
            "cardinality": "auto",
            "likelihood": "gaussian",
            "num_eval_samples": 1000,
            "test_quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
    }

    score_command = ScoringCommand(
        score_name="Testing score command",
        target_column="target1",
        model_name="DEEPAR",
        model_config=score_model_configs,
        score_status="Created",
        prediction_steps=3,
        score_time_frequency="M",
    )
    forecast = deep_ar_modeler.handle(score_command)
    assert forecast is not None
    assert len(forecast) > 0
