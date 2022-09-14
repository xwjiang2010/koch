import datetime
import json
import os
import sys
import time
from pprint import pprint

import numpy as np
import pandas as pd
import pytest
import yaml

##from prediction.src.ensemble import CompetitiveModelsEnsemble

np.random.seed(0)

arima_p = {"arima_params": {"p": 1, "d": 1, "q": 0}}
dt_range = pd.date_range(start="07/1/2017", end="05/01/2020", freq="MS")  ## 36
n = len(dt_range)
values = np.random.uniform(0, 9000000, size=n)
df = pd.DataFrame({"y": values}, index=dt_range)
values_train = values[:30]
