import datetime
import json
import os
import tarfile
import time

import numpy as np
import pandas as pd
import torch
from _nn_base import BaseMLP
from lstm_network import LSTM
from sagemaker_hpo import SageMakerTrainer


class LSTMPredictor(BaseMLP, SageMakerTrainer):
    def __init__(
        self,
        config,
        tuned_job_name=None,
        tunable_parmas_str="tunable_params",
        hyper_params_str="hyper_params",
        source_dir="nn",
        sagemaker_entry_point="lstm_sagemaker.py",
    ):
        super().__init__(config)
        self.job_name = tuned_job_name
        self.source_dir = source_dir
        self.sagemaker_entry_point = sagemaker_entry_point
        self.hyperparam = super().config[hyper_params_str]
        self.tunable = super().config[tunable_parmas_str]
        self.tunable_params = {
            "epochs": IntegerParameter(*self.tunable["epochs"]),
            "context-length": IntegerParameter(*self.tunable["context-length"]),
            "hidden-dim": IntegerParameter(*self.tunable["hidden-dim"]),
            "num-layers": IntegerParameter(*self.tunable["num-layers"]),
            "bias": CategoricalParameter(self.tunable["bias"]),
            "fully-connected": CategoricalParameter(self.tunable["fully-connected"]),
            "lr": ContinuousParameter(*self.tunable["lr"]),
        }
        self.metric_definitions = [
            {
                "Name": "average test loss",
                "Regex": "Test set: Average loss: ([0-9\\.]+)",
            },
            {
                "Name": "average train loss",
                "Regex": "Train set: Train loss: ([0-9\\.]+)",
            },
            {"Name": "total loss", "Regex": "sum_Train_Test loss: ([0-9\\.]+)"},
        ]
        if self.job_name is not None:
            df = sagemaker.HyperparameterTuningJobAnalytics(self.job_name).dataframe()
            best_job_df = df[df["TrainingJobName"] == best_job_name]
            best_job_dict = best_job_df.to_dict(orient="records")[0]

    def fit(self, X, y, input_channels, job_name, output_path):
        """
        Training function for LSTM
        """
        estimator = PyTorch(
            entry_point=self.sagemaker_entry_point,
            role=get_execution_role(),
            source_dir=self.source_dir,
            framework_version="1.0",
            output_path=output_path,
            train_instance_count=super().train_instance_count,
            train_instance_type=super().train_instance_type,
            hyperparameters=self.hyperparam,
            tags=super().tags,
        )
        self.job_name = job_name

        tuner = super._fit_sagemaker(
            estimator=estimator,
            tunable_params=self.tunable_params,
            input_channels=input_channels,
            metric_definitions=self.metric_definitions,
        )

        return self

    def predict(self, X, pred_steps):
        model_path = os.path.join("TODO", "model.pth")
        self._load_model(model_path)
        pred = []
        is_normalize = self.config["hyper_params"]["normalized"]
        for line_record in X:
            if is_normalize:
                mean = self.mean_dict[tuple(line_record["index_cols"])]
                std = self.std_dict[tuple(line_record["index_cols"])]
                record = transform(line_record["target"], mean, std)
            else:
                record = line_record["target"]
            temp = {}
            # rewrite this , prediction do not have y_test
            X_pred = record[-int(self.tunable["context-length"]) :]
            with torch.no_grad():
                X_pt = torch.from_numpy(X_pred).type("torch.FloatTensor")
                y_pred = []
                for _ in range(self.config["no_samples"]):
                    y_pt = model(X_pt.view(-1, *X_pt.size()))
                    y_pred_temp = y_pt.squeeze().numpy()[-1]
                    y_pred.append(y_pred_temp)
            if is_normalize:
                p_vals = inv_transform(y_pred, mean, std)
            else:
                p_vals = y_pred
            p_datetime = (
                pd.Timestamp(line_record["start"], freq=super().frequency)
                + len(line_record["target"])
                - 1
                + super().pred_length
            )
            temp = {
                "index_cols": tuple(line_record["index_cols"]),
                "datetime": p_datetime,
                "prediction": np.nanmean(p_vals),  ## TODO
                "variance": np.nanvar(p_vals),
                "pred_vals": p_vals,
            }
            pred.append(temp)

        return pd.DataFrame(pred)

    def _load_model_params(self):
        # load best job params
        job_name = self.config["tuning_job_name"]
        tuning_results = boto3.client("sagemaker").describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=job_name
        )

        best_job_dict = tuning_results["BestTrainingJob"]["TunedHyperParameters"]
        best_job_name = tuning_results["BestTrainingJob"]["TrainingJobName"]
        if self.config["training_job_name"] is not None:
            best_job_name = m_config["training_job_name"]
            df = sagemaker.HyperparameterTuningJobAnalytics(job_name).dataframe()
            best_job_dict = df[df["TrainingJobName"] == best_job_name].to_dict(
                orient="records"
            )[0]

        # download the model artifact to local output folder
        model_path = os.path.join(
            self.config["S3_PREFIX"],
            "output",
            module_name,
            best_job_name,
            "output",
            "model.tar.gz",
        )
        local_path = os.path.join(self.config["OUTPUT_FOLDER"], "lstm1", "model.tar.gz")
        boto3.resource("s3").Bucket(config["S3_DATA_BUCKET"]).download_file(
            model_path, local_path
        )
        # untar model artifacts to outputfolder
        with tarfile.open(local_path) as tf:
            tf.extractall(path=os.path.join(self.config["OUTPUT_FOLDER"], "lstm1"))

        self.params = best_job_dict
        if self.config["hyper_params"]["normalized"] is True:
            try:
                mean_dict = load_pickle(os.path.join(LSTM1_OUT, "mean.pickle"))
                std_dict = load_pickle(os.path.join(LSTM1_OUT, "std.pickle"))
            except:
                mean_dict, std_dict = get_mean_std(
                    path=os.path.join(LSTM1_OUT, "train.json")
                )

        self.mean_dict = mean_dict
        self.std_dict = std_dict

    def _load_model(self, model_path, is_predict=True):
        self._load_model_params()
        if is_predict:
            dropout = self.params["hyper_params"]["dropout-eval"]
            model = LSTM(
                1,
                int(self.params["hidden-dim"]),
                self.pred_steps,
                int(self.params["context-length"]),
                bias=bool(self.params["bias"]),
                num_layers=int(self.params["num-layers"]),
                fully_connected=bool(self.params["fully-connected"]),
                dropout_eval=float(dropout),
            )
        else:
            model = LSTM(
                1,
                int(self.params["hidden-dim"]),
                self.pred_steps,
                int(self.params["context-length"]),
                bias=bool(self.params["bias"]),
                num_layers=int(self.params["num-layers"]),
                fully_connected=bool(self.params["fully-connected"]),
            )

        self.model = model
        self.model.load_state_dict(torch.load(model_path))

        return self.model
