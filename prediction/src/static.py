import os
from dataclasses import dataclass, field
from typing import Dict

from prediction.src.metaclass_utils import Singleton

# ALLOWS base path to be passed from env for other envs and testing
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "s3://prediction-services/")


@dataclass
class PredictionConfigContext(metaclass=Singleton):
    db_uri = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:local_db_password@postgres-postgresql/postgres",
    )
    kgsa_core_prediction_version: str = os.environ.get("IMAGE_TAG", "")
    S3_PRED_PATH: str = OUTPUT_PATH
    S3_DATA_PATH: str = OUTPUT_PATH
    S3_PRED_MODEL_PATH: str = f"{OUTPUT_PATH}model/"
    S3_SCORE_DATA_PATH: str = f"{OUTPUT_PATH}score/"
    S3_TRAIN_RESULT_PATH: str = f"{OUTPUT_PATH}train/"
    S3_DURABLE_TRIAL_PATH: str = f"{OUTPUT_PATH}durable-trials/"
    logger_name: str = "prediction_services"
    default_loss_function: str = "MAPE"
    default_execution_environment: str = "Sequential"
    DTM_FORMAT: str = r"%Y-%m-%d"
    default_data_storage_type: str = "s3"
    default_file_type: str = "csv"
    default_time_interval: str = "M"
    default_prediction_count: int = 1
    default_training_task: str = "Model"
    default_tuning_algorithm: str = "PBT"
    default_train_job_type: str = "Single-Target"
    default_prediction_steps: int = 3
    default_models_config: Dict = field(init=False)
    default_tune_config: Dict = field(init=False)
    RANDOM_SEED = 0
    ensemble_base_models = ["ARIMA", "HOLTWINTERS", "PROPHET"]
    supported_models = [
        "ARIMA",
        "HOLTWINTERS",
        "PROPHET",
        "ENSEMBLE",
        "DEEPAR",
        "LSTM",
        "MOVINGAVERAGE",
    ]

    def __post_init__(self):
        self.default_models_config = {}

        self.default_models_config["LSTM"] = {
            "parameters": {
                "FREQ": "MS",
                "train_instance_count": 1,
                "train_instance_type": "ml.m4.10xlarge",  # ml.c5.4xlarge
                "job_name_prefix": "",  # update
                "early_stopping_type": "Auto",
                "max_jobs": 30,
                "max_parallel_jobs": 1,
                "tunable_params": {
                    "epochs": 300,
                    "context-length": 6,
                    "lr": 0.001,
                    "hidden-dim": 80,
                    "num-layers": 2,
                    "bias": True,
                    "fully-connected": True,
                },
            },
            "hyperparameters": {
                "batch_size": 10000,
                "test-batch-size": 10000,
                "log-interval": 1000,
                "dropout-eval": 0.2,
                "normalized": True,
                "test-loss-func": "mape",
                "loss-func": "mape",
            },
        }

        self.default_models_config["ARIMA"] = {
            "parameters": {"p": 1, "d": 1, "q": 0},
            "hyperparameters": {"disp": 0},
        }
        self.default_models_config["HOLTWINTERS"] = {
            "parameters": {"trend": "add", "damped": True, "seasonal": None},
            "hyperparameters": {
                "optimized": True,
                "use_boxcox": True,
                "remove_bias": True,
            },
        }
        self.default_models_config["PROPHET"] = {
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
        self.default_models_config["DEEPAR"] = {
            "parameters": {
                "mini_batch_size": [
                    10000,
                ],
                "epochs": [
                    50,
                ],
                "context_length": [
                    2,
                ],
                "num_cells": [
                    15,
                ],
                "num_layers": [
                    2,
                ],
                "dropout_rate": [
                    0.02,
                ],
                "embedding_dimension": [
                    10,
                ],
                "learning_rate": [
                    0.001,
                ],
            },
            "hyperparameters": {
                "time_freq": "M",
                "early_stopping_patience": 5,
                "cardinality": "auto",
                "likelihood": "gaussian",
                "num_eval_samples": 1000,
                "test_quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            },
            "sagemaker_params": {
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
                "iam_role": os.environ.get(
                    "AWS_IAM_ROLE",
                    "arn:aws:iam::254486207130:role/kbs-analytics-dev-fhr-demand-forecasting-SageMaker-ExecutionRoll",
                ),
                "tags": [
                    {"Key": "Name", "Value": "kgsalasercats@kochind.onmicrosoft.com"},
                    {"Key": "costcenter", "Value": "54751"},
                    {"Key": "blc", "Value": "1559"},
                ],
            },
        }

        self.default_models_config["MOVINGAVERAGE"] = {
            "parameters": {},
            "hyperparameters": {"number_of_periods": 0, "imputation": True},
        }

        self.default_tune_config = {}
        self.default_tune_config["ARIMA"] = {
            "parameters": {
                "p": [1, 2, 3, 4, 5],  # 6, 7],
                "d": [0, 1],
                "q": [0, 1, 2],  # 3],
            },
            "hyperparameters": {"disp": 0},
        }
        self.default_tune_config["HOLTWINTERS"] = {
            "parameters": {
                "trend": ["add", "mul", None],
                "damped": [True, False],
                # "seasonal_periods": [2, 3, 4, 6, 12],
                "seasonal": ["add", "mul", None],
            },
            ##"hyperparameters": {"smoothing_level": [0.1, 0.2, 0.4, 0.8]},
            "hyperparameters": {
                "optimized": True,
                "use_boxcox": True,
                "remove_bias": True,
            },
        }
        self.default_tune_config["LSTM"] = {
            "parameters": {
                "FREQ": "MS",
                "train_instance_count": 1,
                "train_instance_type": "ml.m4.10xlarge",  # ml.c5.4xlarge
                "job_name_prefix": "",  # update
                "early_stopping_type": "Auto",
                "max_jobs": 30,
                "max_parallel_jobs": 1,
                "tunable_params": {
                    "epochs": [50, 300],
                    "context-length": [4, 6],
                    "lr": [0.001, 0.01],
                    "hidden-dim": [10, 100],
                    "num-layers": [1, 2],
                    "bias": [True, False],
                    "fully-connected": [True, False],
                },
            },
            "hyperparameters": {
                "batch_size": 10000,
                "test-batch-size": 10000,
                "log-interval": 1000,
                "dropout-eval": 0.2,
                "normalized": True,
                "test-loss-func": "mape",
                "loss-func": "mape",
            },
        }
        self.default_tune_config["PROPHET"] = {
            "parameters": {
                "growth": ["linear"],  ##, "logistic"],
                "changepoints": [None],
                "n_changepoints": [20, 22],
                "changepoint_range": [0.8, 0.9],  ##0.6, 0.7, 0.8, 0.9],
                "changepoint_prior_scale": [
                    0.05,
                    0.1,
                ],  ##[0.001, 0.01, 0.05, 0.1, 0.5],
                "yearly_seasonality": ["auto"],
                "weekly_seasonality": ["auto"],
                "daily_seasonality": ["auto"],
                "holidays": [None],
                "seasonality_mode": ["additive", "multiplicative"],
                "seasonality_prior_scale": [10.0],
                "holidays_prior_scale": [10.0],
                "interval_width": [0.8],
                "uncertainty_samples": [1000],
            },
            "hyperparameters": {},
        }
        self.default_tune_config["DEEPAR"] = {
            "parameters": {
                "mini_batch_size": [
                    10000,
                ],
                "epochs": [30, 300],
                "context_length": [3, 10],
                "num_cells": [1, 20],
                "num_layers": [1, 5],
                "dropout_rate": [0.0, 0.10],
                "embedding_dimension": [1, 15],
                "learning_rate": [1.0e-3, 1.0e-2],
            },
            "hyperparameters": {
                "time_freq": "M",
                "early_stopping_patience": 5,
                "cardinality": "auto",
                "likelihood": "gaussian",
                "num_eval_samples": 1000,
                "test_quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            },
            "sagemaker_params": {
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
                "iam_role": os.environ.get(
                    "AWS_IAM_ROLE",
                    "arn:aws:iam::254486207130:role/kbs-analytics-dev-fhr-demand-forecasting-SageMaker-ExecutionRoll",
                ),
                "tags": [
                    {"Key": "Name", "Value": "kgsalasercats@kochind.onmicrosoft.com"},
                    {"Key": "costcenter", "Value": "54751"},
                    {"Key": "blc", "Value": "1559"},
                ],
            },
        }
        self.default_tune_config["MOVINGAVERAGE"] = {
            "parameters": {},
            "hyperparameters": {"number_of_periods": 0, "imputation": True},
        }


PREDICTION_CONTEXT = PredictionConfigContext()
