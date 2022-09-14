"""
Base class for ARIMA estimator
"""

from datetime import datetime

from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima_model import ARIMA

from prediction.src.ts_base import TsMixin

from ._tsa_base import UnivariateTsPredictor


class ARIMAPredictor(UnivariateTsPredictor, TsMixin):
    def __init__(self, config, model_params_str="arima_params", alpha=0.2):
        super().__init__(config, model_params_str)
        self.alpha = alpha
        self.model_params = config[model_params_str]
        self.p_value = self.model_params["p"]
        self.d_value = self.model_params["d"]
        self.q_value = self.model_params["q"]
        self.estimator_params = (self.p_value, self.d_value, self.q_value)
        self.model_fit = None
        self.model = None
        self.out_sample_features = 0

        # State-space model requires:
        # 'enforce_stationarity', 'enforce_invertibility','concentrate_scale'

    def fit(self, X, y, features=None):
        """
        Fit or train ARIMA model on the given data
        Evaluate parameters on the data
        Instantiate ARIMA model and fit on the data
        """
        history = y
        exog = None
        """
        Instantiate ARIMA model and fit on the data
        """
        history = history.astype("float32")

        if features is not None and len(features) > 0:
            exog = features.copy()
            self.out_sample_features = max(0, len(exog.index) - len(y))
            exog.drop(exog.tail(self.out_sample_features).index, inplace=True)
        else:
            exog = None
            self.out_sample_features = 0

        self.model = ARIMA(endog=history, exog=exog, order=self.estimator_params)
        ## TODO dates = dts, freq = self.freq)

        model_fit = self.model.fit(alpha=self.alpha)
        self.model_fit = model_fit
        return self

    def predict(
        self,
        X,
        y,
        features=None,
        steps=3,
        prediction_count=3,
        clip_negative=0,
        weights=None,
    ):
        """
        Predict steps ahead
        X: historical data
        """
        if not self.model_fit:
            self.fit(X, y, features)

        if features is not None and len(features) > 0:
            features = features.tail(len(features.index) - len(y))
            features = features.head(steps)

            if len(features.index) != steps:
                raise ValueError(
                    f"There are not enough external(exog) features  to support this prediction. There must be enough external features {len(features)} for each step: {steps}"
                )
            # if list(features.index)[0] != (list(X)[-1] + self.calculate_next_ts_interval(list(X))):
            #     raise ValueError(
            #         f"The first Date in the features set is not the next expected date in a continouse time series. Something is wrong"
            #     )
        else:
            features = None

        forecast = self.model_fit.forecast(steps=steps, exog=features)

        prediction = forecast[0][-prediction_count:]
        pred = UnivariateTsPredictor.remove_negative(prediction, clip_negative)

        return pred
