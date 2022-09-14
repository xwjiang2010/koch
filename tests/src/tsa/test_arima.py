import datetime
import json
import os
import sys
import time
import unittest
from pprint import pprint

import numpy as np
import pandas as pd
import pytest
import yaml

from prediction.src.tsa.arima_predictor import ARIMAPredictor

np.random.seed(0)


class ARIMAPredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(self, *args, **kwargs):
        self.arima_p = {"arima_params": {"p": 1, "d": 1, "q": 0}}
        self.dt_range = pd.date_range(
            start="07/1/2017", end="05/01/2020", freq="MS"
        )  ## 36
        n = len(self.dt_range)
        self.values = np.random.uniform(0, 9000000, size=n)
        self.df = pd.DataFrame({"y": self.values}, index=self.dt_range)
        self.values_train = self.values[:30]
        self.arima = ARIMAPredictor(config=self.arima_p)

    def testNormalFit(self):
        arima_fit = self.arima.fit(self.dt_range, self.values)
        self.assertIsNotNone(arima_fit.model_fit)
        model = arima_fit.model_fit
        summary = model.summary()
        self.assertIsNotNone(summary)
        # Note that tables is a list. The table at index 1 is the "core" table.
        # Additionally, read_html puts dfs in a list, so we want index 0
        results_as_html = summary.tables[1].as_html()
        self.assertIsNotNone(results_as_html)
        df1 = pd.read_html(results_as_html, header=0, index_col=0)[0]
        self.assertIsNotNone(df1)

    def testNormalPredict(self, pred_count=2):
        forecast = self.arima.predict(self.dt_range, self.values, steps=pred_count)
        self.assertIsNotNone(forecast)
        self.assertGreater(len(forecast), 0)
        self.assertEqual(len(forecast), pred_count)

    def testNormalScore(self, pred_count=2):
        score = self.arima.score(self.dt_range, self.values, steps=pred_count)

        self.assertIsNotNone(score)
        self.assertGreater(score, -100)


if __name__ == "__main__":
    HOME_PATH = "/home/ec2-user/SageMaker/ts_prediction/"
    sys.path.append(HOME_PATH)
    sys.path.append(os.path.join(HOME_PATH, "metrics"))
    sys.path.append(os.path.join(HOME_PATH, "tsa"))
    sys.path.append(os.path.join(HOME_PATH, "utils"))
    sys.path.append(os.path.join(HOME_PATH, "nn"))
