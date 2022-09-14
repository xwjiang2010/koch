import inspect
from abc import abstractmethod
from itertools import compress
from typing import Optional, Union

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from prediction.src.metrics.regression_metrics import (
    adjusted_r2_score,
    calc_bias,
    calc_mape,
    rmse,
)

"""
Base class for all estimators
"""


class TsPredictor:
    """
    Base class for all time series estimators

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "KGS estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                warnings.warn(
                    "get_params will raise an "
                    "AttributeError if a parameter cannot be "
                    "retrieved as an instance attribute. Previously "
                    "it would return None.",
                    FutureWarning,
                )
                value = None
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : object
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    @abstractmethod
    def fit(self, X, y):
        """
        Fit or train model
        """

    @abstractmethod
    def predict(self, X):
        """
        Predict on new data
        """


class TsMixin:
    """
    Mixin class for all  traditional time series estimators:
    1. ARIMA
    2. HoltWinter
    3. Prophet
    """

    _error_dict = {
        "MAPE": calc_mape,
        "R2": r2_score,
        "MAE": mean_absolute_error,
        "MEDIANAE": median_absolute_error,
        "MSE": mean_squared_error,
        "RMSE": rmse,
        "AR2": adjusted_r2_score,
    }

    def _get_prediction(self, X, y, steps=3, features=None, sample_weights=1):
        y_copy = y.copy()
        features = features.copy() if features is not None else None
        # dfFeature_valid = self.validationFeature_data.copy() if self.validationFeature_data is not None else None

        y_hat = self.predict(X, y, steps=steps, features=features)
        n_window = len(y_hat)
        if isinstance(y, pd.DataFrame):
            valid = y_copy[-n_window:].squeeze()
        else:
            valid = y_copy[-n_window:]
        if steps == 1:
            valid = [valid]

        assert len(valid) == len(y_hat), f"Valid data: {valid} \n y_hat: {y_hat}"
        return (valid, y_hat)

    def score(
        self,
        X: list,
        y: list,
        sample_weights=None,
        error_function: str = "MAPE",
        steps: int = 3,
        features: Optional[list] = None,
    ):
        """
        Return the error of the prediction
        """

        if isinstance(error_function, str):
            if error_function in self._error_dict:
                error_function = self._error_dict[error_function]
            else:
                raise ValueError(
                    "Provided error_function {} is not recongized. Only following error functions are available: {}".format(
                        error_function, self._error_dict.values()
                    )
                )

        valid, y_hat = self._get_prediction(
            X=X, y=y, steps=steps, sample_weights=sample_weights, features=features
        )
        score = error_function(valid, y_hat)

        return score

    def score_iteratively(
        self,
        X: list,
        y: list,
        sample_weights= None,
        error_function: str = "MAPE",
        steps: int = 3,
        prediction_count: int = 3,
        exog: Optional[list] = None,
    ):
        """
        Return the error of the prediction by stepping one interval at a time
        """

        score = 0
        for window in range(1, steps + 1):
            # train
            history = y[:-window]
            input = X[:-window]
            self.fit(input, history, features=exog)
            curr_score = self.score(
                X,
                y,
                sample_weights=sample_weights,
                error_function=error_function,
                steps=1,
                features=exog,
            )

            score = score + curr_score

        score = score / steps

        return score

    def calc_bias(
        self,
        X: list,
        y: list,
        prediction_sample_weights=1,
        sample_weights= 1,
        steps: Optional[int] = 3,
        features: Optional[list] = None,
    ):

        # We make the assumption get predictions will return
        # a consistent result, with LRU cache it should.
        valid, y_hat = self._get_prediction(
            X=X,
            y=y,
            steps=steps,
            sample_weights=prediction_sample_weights,
            features=features,
        )

        return calc_bias(valid, y_hat, sample_weights)
