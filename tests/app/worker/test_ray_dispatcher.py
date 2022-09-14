import argparse
import json
import logging
import multiprocessing as mp
import os
import pprint
import sys
import time
import unittest
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
import ray
from dateutil.relativedelta import *
from pony.orm import *

# from ray.cluster_utils import Cluster
from tabulate import tabulate

##import prediction.src.tune.ray_tune_utils as ray_tune
import prediction.src.db.prediction_database as pred_db
import prediction.src.s3_files as s3_files
from prediction.app.prediction_commands import TrainingModel
from prediction.app.worker.training_handlers import TSAModeler
from prediction.src.db.prediction_database import (
    TrainRunConfig,
    TrainRunModelConfig,
    create_train_run_config,
    get_train_run_config,
    get_train_run_config_by_uuid,
)
from prediction.src.static import PREDICTION_CONTEXT

logger = logging.getLogger(PREDICTION_CONTEXT.logger_name)

S3_PRED_PATH = "tests/resources/data/"


@pytest.fixture
def ray_cluster():
    # Starts a head-node for the cluster.
    cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 10,
        },
    )
    yield cluster


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
        },
    }


## TODO test actual
@pytest.mark.skip
def testDispatchTrain(ray_cluster, train_run_config):
    config = create_train_run_config(train_run_config)
    assert config is not none
