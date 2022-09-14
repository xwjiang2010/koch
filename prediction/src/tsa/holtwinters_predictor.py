"""
Base class for HoltWinter and ExponentialSmoothing estimator
"""

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from prediction.src.ts_base import TsMixin

from ._tsa_base import UnivariateTsPredictor


class HoltWintersPredictor(UnivariateTsPredictor, TsMixin):
    def __init__(
        self, config, model_params_str="holt_params", hyperparams_str="holt_hyperparams"
    ):
        super().__init__(config, model_params_str)
        ##self.model_config = config[model_params_str]
        self.model_params = config[model_params_str]
        self.trend = self.model_params["trend"]
        self.damped = self.model_params["damped"]
        self.seasonal = self.model_params["seasonal"]
        self.estimator_params = [self.trend, self.damped, self.seasonal]
        self.fit_params = config[hyperparams_str]
        # TODO - Using other params
        self.seasonal_periods = self.model_params.get("seasonal_periods", None)
        self.initialization_method = self.model_params.get(
            "initialization_method", None
        )
        self.use_boxcox = self.model_params.get("use_boxcox", False)
        self.model_fit = None

    def fit(self, X, y, features=None):
        """
        Evaluate parameters on the data
        """
        ## TODO: Comment about X being unnecessary
        """
        Instantiate ExponentialSmoothing model and fit on the data
        """
        history = y.astype("float32")
        self.model = ExponentialSmoothing(history, **self.model_params)
        ## TODO , freq=self.freq)
        self.model_fit = self.model.fit(**self.fit_params)
        return self

    def predict(
        self, X, y, steps=3, prediction_count=3, clip_negative=0, features=None
    ):
        """
        Predict steps ahead
        X: historical data
        """
        if not self.model_fit:
            self.fit(X=None, y=y)
        forecast = self.model_fit.forecast(steps=steps)
        prediction = forecast[-prediction_count:]
        pred = UnivariateTsPredictor.remove_negative(prediction, clip_negative)

        return pred
