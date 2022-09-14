import copy
import logging

import prediction.app.worker.ray_processors as ray_proc
from prediction.app.prediction_commands import HyperparameterTuning

logger = logging.getLogger("prediction_services")

arima = {"p": [1, 7], "d": [1, 2], "q": [1, 2]}
holt = {
    "trend": ["add", "mul"],
    "damped": [True, False],
    "seasonal_periods": 3,
    "seasonal": [None],
}
prophet = {
    "growth": ["linear"],
    "changepoints": [None],
    "n_changepoints": [20, 22],
    "changepoint_range": [0.8, 0.9],
    "changepoint_prior_scale": [0.05, 0.1],
    "yearly_seasonality": ["auto"],
    "weekly_seasonality": ["auto"],
    "daily_seasonality": ["auto"],
    "holidays": [None],
    "seasonality_mode": ["additive", "multiplicative"],
    "seasonality_prior_scale": [10.0],
    "holidays_prior_scale": [10.0],
    "interval_width": [0.8],
    "uncertainty_samples": [1000],
}

trainCmd = HyperparameterTuning(
    train_model_id="1",
    train_name="Test-RayTuneConfigCreator",
    train_description="Testing ray configuration creation",
    train_job_type="Sequential",
    metrics={},
    model_location="s3://",
    model_name="ARIMA",
    model_config=None,
    model_metric="MAPE",
    model_time_frequency="M",
    model_params={},
    model_hyperparams={},
    hyperparam_algorithm="GRID-SEARCH",
    modeler=None,
    data_preprocessor=None,
)
rayConfig = ray_proc.RayTuneConfigCreator()


def testRayTuneConfigCreatorArima():
    arimaCmd = copy.deepcopy(trainCmd)
    arimaCmd.model_params = arima
    tuneParams = rayConfig.handle(arimaCmd)
    assert tuneParams is not None
    ##logger.warn(f'p: {tuneParams["p"]}')
    assert ("p" in tuneParams) == True
    assert ("d" in tuneParams) == True
    assert ("q" in tuneParams) == True


def testRayTuneConfigCreatorHoltwinter():
    holtCmd = copy.deepcopy(trainCmd)
    holtCmd.model_name = "HOLTWINTERS"
    holtCmd.model_params = holt
    tuneParams = rayConfig.handle(holtCmd)
    assert tuneParams is not None
    logger.warn(f"tuneParams: {tuneParams}")
    assert ("trend" in tuneParams) == True
    assert ("damped" in tuneParams) == True
    assert ("seasonal" in tuneParams) == True


def testRayTuneConfigCreatorProphet():
    prophetCmd = copy.deepcopy(trainCmd)
    prophetCmd.model_name = "PROPHET"
    prophetCmd.model_params = prophet
    tuneParams = rayConfig.handle(prophetCmd)
    assert tuneParams is not None
    logger.warn(f"tuneParams: {tuneParams}")
    assert ("growth" in tuneParams) == True
    assert ("n_changepoints" in tuneParams) == True
    assert ("holidays" in tuneParams) == True
