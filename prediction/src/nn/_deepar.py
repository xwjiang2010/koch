import pandas as pd
from dateutil.relativedelta import *

from ._nn_base import BaseMLP
from .sagemaker_hpo import SageMakerTrainer


class DeepARPredictor(BaseMLP):
    def __init__(
        self,
        config,
        tuned_job_name=None,
        tunable_params_str="tunable_params",
        hyper_params_str="hyper_params",
        trainer=None,
    ):
        self.num_samples = config.get("num_samples", 1000)
        self.hyperparam = config[hyper_params_str]
        self.hyperparameters = {
            "time_freq": self.hyperparam["time_freq"],
            "early_stopping_patience": str(self.hyperparam["early_stopping_patience"]),
            "prediction_length": str(config["pred_steps"]),
            "cardinality": self.hyperparam["cardinality"],
            "likelihood": self.hyperparam["likelihood"],
            "num_eval_samples": self.hyperparam["num_eval_samples"],
            "test_quantiles": self.hyperparam["test_quantiles"],
        }
        self.tunable = config[tunable_params_str]
        self.job_name = tuned_job_name
        self.endpoint_name = config.get("endpoint_name", None)
        if self.endpoint_name:
            self.predictor = Predictor(self.endpoint_name)
        self.trainer = trainer
        if not self.trainer:
            config["job_name"] = self.job_name
            config["num_samples"] = self.num_samples
            self.trainer = SageMakerTrainer(config, self.tunable, self.hyperparameters)

    def fit(self, X, y, input_channels, job_name, output_path):
        """
        training function for deepar
        """
        # TODO: X and y are unnecessary
        if job_name:
            self.job_name = job_name

        status = self.trainer.fit(X, y, input_channels, self.job_name, output_path)
        return status

    def predict(self, X, pred_steps):
        pred = []
        if pred_steps is None:
            pred_steps = self.pred_length
        for idx, line_record in enumerate(X):
            temp = {}
            p = self.trainer.predict_on_request(line_record, pred_steps)
            p_median = p["predictions"][0]["quantiles"]["0.5"][-1]
            p_10 = p["predictions"][0]["quantiles"]["0.1"][-1]
            p_90 = p["predictions"][0]["quantiles"]["0.9"][-1]
            p_mean = p["predictions"][0]["mean"][-1]
            p_vals = [s[-1] for s in p["predictions"][0]["samples"]]
            assert len(p_vals) == self.num_samples
            time_delta = len(line_record["target"]) - 1 + pred_steps
            p_datetime = pd.Timestamp(
                line_record["start"], freq=self.trainer.frequency
            ) + relativedelta(months=time_delta)
            cat = line_record["index_cols"]
            temp = {
                "index_cols": tuple(cat),
                "datetime": p_datetime,
                "prediction": p_mean,
                "prediction_50": p_median,
                "prediction_10": p_10,
                "prediction_90": p_90,
            }
            pred.append(temp)

        return pd.DataFrame(pred)
