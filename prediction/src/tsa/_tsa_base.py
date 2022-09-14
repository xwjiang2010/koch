"""
Time series analysis base params and functions
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from pmdarima.arima.utils import ndiffs
from pmdarima.utils import diff

from prediction.src.ts_base import TsPredictor


##@ray.remote
class UnivariateTsPredictor(TsPredictor, metaclass=ABCMeta):
    def __init__(self, config, model_params_str):
        self.config = config
        self.model_params = config[model_params_str]
        self.freq = config.get("freq", "M")
        ##self.error_function = error
        ##self.clip_negative = clip_negative
        ##self.n_process = n_process

    @staticmethod
    def remove_negative(prediction, clip_negative=0):
        if clip_negative is not None:
            # clip negative predictions to zero
            ##pred = np.maximum(clip_negative, prediction)
            pred = [np.maximum(clip_negative, i) for i in prediction]

        return pred

    @staticmethod
    def detrend_data(target_df, exog_df):
        exog_cols = list(exog_df.columns)

        # Estimate the number of differences using an ADF test:
        exog_dict = {}
        for curr_col in exog_cols:
            curr_diff_n = ndiffs(exog_df[curr_col], test="adf")
            exog_dict[curr_col] = curr_diff_n

        # Determine max diff
        max_diff = max(exog_dict.values())

        # Difference each series corresponding to its difference count
        exog_diff_dict = {}
        for curr_col in exog_cols:
            curr_diff_n = exog_dict[curr_col]
            curr_diff_val = exog_df[curr_col]
            if curr_diff_n > 0:
                curr_diff_val = diff(exog_df[curr_col], differences=curr_diff_n)

            row_start = max_diff - curr_diff_n
            curr_exog = curr_diff_val[row_start:]
            exog_diff_dict[curr_col] = curr_exog

        exog_diff_df = pd.DataFrame(exog_diff_dict)
        target_n = target_df.shape[0]
        diff_n = exog_diff_df.shape[0]
        target_start = target_n - diff_n

        target_diff_df = target_df[target_start:]

        return target_diff_df, exog_diff_df, exog_dict

    @abstractmethod
    def fit(self, X, y, features):
        """
        Fit or train model
        """
