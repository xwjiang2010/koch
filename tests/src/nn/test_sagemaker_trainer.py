import copy
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import unittest
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import ray

import prediction.src.db.prediction_database as pred_db
import prediction.src.nn.sagemaker_hpo as sagemaker_manager
from prediction.src.static import PREDICTION_CONTEXT

logger = logging.getLogger(PREDICTION_CONTEXT.logger_name)

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
    "iam_role": "arn:aws:iam::254486207130:role/kbs-analytics-dev-fhr-demand-forecasting-SageMaker-ExecutionRoll",
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


def test_job_handler():
    mgr = sagemaker_manager.SageMakerTrainer(
        deepar_p, deepar_p["tunable_params"], deepar_p["hyper_params"]
    )
    assert mgr is not None
    config = mgr.config
    assert config is not None
    tunable = mgr.tunable
    assert tunable is not None
    hyperparamters = mgr.hyperparameters
    assert hyperparamters is not None


# get_job_status
def test_status_of_job():
    with unittest.mock.patch(
        "prediction.src.nn.sagemaker_hpo.SageMakerTrainer.get_job_status",
        return_value="InProgress",
    ):
        mgr = sagemaker_manager.SageMakerTrainer(
            deepar_p, deepar_p["tunable_params"], deepar_p["hyper_params"]
        )
        assert mgr is not None
        assert mgr.get_job_status() == "InProgress"

    with unittest.mock.patch(
        "prediction.src.nn.sagemaker_hpo.SageMakerTrainer.get_job_status",
        return_value="Completed",
    ):
        mgr = sagemaker_manager.SageMakerTrainer(
            deepar_p, deepar_p["tunable_params"], deepar_p["hyper_params"]
        )
        assert mgr is not None
        assert mgr.get_job_status() == "Completed"

    with unittest.mock.patch(
        "prediction.src.nn.sagemaker_hpo.SageMakerTrainer.get_job_status",
        return_value="Stopped",
    ):
        mgr = sagemaker_manager.SageMakerTrainer(
            deepar_p, deepar_p["tunable_params"], deepar_p["hyper_params"]
        )
        assert mgr is not None
        assert mgr.get_job_status() == "Stopped"


# check_for_progress
def test_check_for_progress():
    with unittest.mock.patch(
        "prediction.src.nn.sagemaker_hpo.SageMakerTrainer.check_for_progress",
        return_value=True,
    ):
        mgr = sagemaker_manager.SageMakerTrainer(
            deepar_p, deepar_p["tunable_params"], deepar_p["hyper_params"]
        )
        assert mgr is not None
        assert mgr.check_for_progress() == True

    with unittest.mock.patch(
        "prediction.src.nn.sagemaker_hpo.SageMakerTrainer.check_for_progress",
        return_value=False,
    ):
        mgr = sagemaker_manager.SageMakerTrainer(
            deepar_p, deepar_p["tunable_params"], deepar_p["hyper_params"]
        )
        assert mgr is not None
        assert mgr.check_for_progress() == False


# check_for_progress_handle
def test_check_for_progress_handle():
    with unittest.mock.patch(
        "prediction.src.nn.sagemaker_hpo.SageMakerTrainer.check_for_progress_handle",
        return_value="InProgress",
    ):
        mgr = sagemaker_manager.SageMakerTrainer(
            deepar_p, deepar_p["tunable_params"], deepar_p["hyper_params"]
        )
        assert mgr is not None
        assert mgr.check_for_progress_handle() == "InProgress"

    with unittest.mock.patch(
        "prediction.src.nn.sagemaker_hpo.SageMakerTrainer.check_for_progress_handle",
        return_value="NotInProgress",
    ):
        mgr = sagemaker_manager.SageMakerTrainer(
            deepar_p, deepar_p["tunable_params"], deepar_p["hyper_params"]
        )
        assert mgr is not None
        assert mgr.check_for_progress_handle() == "NotInProgress"


# stop_running_process
def test_stop_running_process():
    with unittest.mock.patch(
        "prediction.src.nn.sagemaker_hpo.SageMakerTrainer.stop_running_process",
        return_value="STOPPED",
    ):
        mgr = sagemaker_manager.SageMakerTrainer(
            deepar_p, deepar_p["tunable_params"], deepar_p["hyper_params"]
        )
        assert mgr is not None
        Completed_Job = mgr.stop_running_process()
        assert Completed_Job == "STOPPED"

    with unittest.mock.patch(
        "prediction.src.nn.sagemaker_hpo.SageMakerTrainer.stop_running_process",
        return_value="SUCCESS",
    ):
        mgr = sagemaker_manager.SageMakerTrainer(
            deepar_p, deepar_p["tunable_params"], deepar_p["hyper_params"]
        )
        assert mgr is not None
        Running_Job = mgr.stop_running_process()
        assert Running_Job == "SUCCESS"
