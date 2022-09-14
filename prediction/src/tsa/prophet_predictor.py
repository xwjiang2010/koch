import os

import pandas as pd
from fbprophet import Prophet
from loguru import logger

from prediction.src.ts_base import TsMixin

from ._tsa_base import UnivariateTsPredictor


class ProphetPredictor(UnivariateTsPredictor, TsMixin):
    def __init__(self, config, model_params_str="prophet_params", mcmc_samples=1000):
        super().__init__(config, model_params_str)
        self.model_params = config[model_params_str]
        self.mcmc_samples = mcmc_samples
        self.model_fit = None
        # self.model_params = config['prophet']
        self.pred_columns = [
            "datetime",
            "test_prediction",
            "test_prediction_10",
            "test_prediction_90",
        ]
        self.forecast_columns = ["ds", "yhat", "yhat_lower", "yhat_upper"]

    def fit(self, X, y, features=None):
        train_data = pd.DataFrame({"ds": X, "y": y})
        self.model = Prophet(mcmc_samples=self.mcmc_samples, **self.model_params)

        # incorporate exteran features into the train data
        if features is not None and len(features) > 0:
            feature_names = features.columns
            features = features[
                features.index <= max(X)
            ]  # trim any out of sample (used for prediction) features
            features = (
                features.reset_index()
            )  # pull Date out of the index (just makes the merge easier)
            train_data = pd.merge(
                train_data, features, left_on="ds", right_on="Date", how="left"
            )
            train_data.drop(columns=["Date"], inplace=True)
            for feature_name in feature_names:
                self.model.add_regressor(feature_name)
            logger.debug(
                f"Prophet adding {len(feature_names)} exogenous features to training"
            )
        with suppress_stdout_stderr():
            self.model_fit = self.model.fit(train_data)
        return self

    def predict(self, X, y, steps=3, prediction_count=1, frequency="MS", features=None):
        if not self.model_fit:
            self.fit(X=X, y=y, features=features)
        future = self.model.make_future_dataframe(periods=steps, freq=frequency)

        if features is not None and len(features) > 0:
            features = features.reset_index()
            future = pd.merge(
                future, features, left_on="ds", right_on="Date", how="left"
            )
            future.drop(columns=["Date"], inplace=True)

        with suppress_stdout_stderr():
            forecast = self.model.predict(future)
        # logger.debug(f"Prophet forecast for {prediction_count}")
        # logger.debug(forecast)
        prediction = forecast[self.forecast_columns].iloc[-prediction_count:]
        # logger.debug(f"********Prophet prediction for {prediction_count}")
        # logger.debug(prediction)
        return prediction.yhat.tolist()


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
