import copy
from asyncio import run_coroutine_threadsafe
from datetime import datetime
from doctest import run_docstring_examples

# from pickle import NONE
from time import sleep
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi import Header
from fastapi.testclient import TestClient

from prediction.master_server import PredictionMediator as PM
from prediction.master_server import app
from prediction.src.db.prediction_database import TrainRunModelConfig
from prediction.src.static import PREDICTION_CONTEXT

# from pyexpat import model


client = TestClient(app)


class PredictionMediatorSpec(PM):
    scoreConfig = SimpleNamespace(
        **{
            "score_run_uuid": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
            "score_run_id": "10",
            "score_status": "Submitted",
        }
    )
    runConfig = SimpleNamespace(
        **{
            "train_run_uuid": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
            "train_run_id": "10",
            "train_status": "Submitted",
        }
    )


###############################################################
#                       Predictions                           #
###############################################################
train_db_result = SimpleNamespace(
    **{
        "train_run_uuid": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "train_status": "Completed",
        "train_data_location": "tests/resources/data/score/model-minimum-config-postman_2021-01-05-18.29.16.csv",
        "score_data_location": "tests/resources/data/score/model-minimum-config-postman_2021-01-05-18.29.16.csv",
        "train_start_ts": datetime.fromisoformat("2021-01-05T18:29:16.246159"),
        "train_end_ts": datetime.fromisoformat("2021-01-05T18:55:12.984947"),
        "update_ts": datetime.fromisoformat("2021-01-05T18:55:12.984947"),
    }
)

### Test Run Training ###
# Fixture for training tests.


@pytest.fixture(scope="function")
def training_post_body():

    return {
        "models": {},
        "training": {
            "train_name": "model-minimum-config-postman",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": [],
            "prediction_steps": 0,
            "metrics": [
                {
                    "name": "bias",
                    "config": [1, 0, 0, 0, 0],
                }
            ],
        },
    }


@pytest.fixture(scope="function")
def exog_training_post_body(training_post_body):
    training_post_body["training"]["model_names"] = ["ARIMA"]
    training_post_body["training"][
        "feature_data_location"
    ] = "tests/resources/data/test_single_exog.csv"
    training_post_body["training"][
        "target_feature_mapping_location"
    ] = "tests/resources/data/test_single_exog_map.csv"
    return training_post_body


# All values in the models match defaults except parameters->d
@pytest.fixture(scope="function")
def arima_training_post_body(training_post_body):

    training_post_body["training"]["model_names"] = ["ARIMA"]

    training_post_body["models"] = {
        "ARIMA": {
            "model_name": "ARIMA",
            "model_time_interval": "M",
            "model_config": {
                "parameters": {
                    "p": [1, 2, 3, 4, 5],
                    "d": [1],
                    "q": [0, 1, 2],
                },
                "hyperparameters": {"disp": 0},
            },
        }
    }
    return training_post_body


@pytest.fixture(scope="function")
def holtwinters_training_post_body(training_post_body):
    training_post_body["training"]["model_names"] = ["HOLTWINTERS"]

    training_post_body["models"] = {
        "HOLTWINTERS": {
            "model_name": "HOLTWINTERS",
            "model_time_interval": "M",
            "model_config": {
                "parameters": {
                    "trend": ["add", "mul", None],
                    "damped": [True, False],
                    "seasonal": ["add", "mul", None],
                },
                "hyperparameters": {
                    "optimized": True,
                    "use_boxcox": True,
                    "remove_bias": True,
                },
            },
        }
    }
    return training_post_body


@pytest.fixture(scope="function")
def prophet_training_post_body(training_post_body):
    training_post_body["training"]["model_names"] = ["PROPHET"]

    training_post_body["models"] = {
        "PROPHET": {
            "model_name": "PROPHET",
            "model_time_interval": "M",
            "model_config": {
                "parameters": {
                    "growth": ["linear"],
                    "changepoints": [None],
                    "n_changepoints": [20],
                    "changepoint_range": [0.8, 0.9],
                    "changepoint_prior_scale": [0.05, 0.1],
                    "yearly_seasonality": ["auto"],
                    "weekly_seasonality": ["auto"],
                    "daily_seasonality": ["auto"],
                    "seasonality_mode": [
                        "additive",
                        "multiplicative",
                    ],
                    "holidays": {},
                    "seasonality_prior_scale": [10.0],
                    "holidays_prior_scale": [10.0],
                    "interval_width": [0.8],
                    "uncertainty_samples": [1000],
                },
                "hyperparameters": {},
            },
        }
    }
    return training_post_body


@patch("prediction.master_server.PredictionMediator", autospec=PredictionMediatorSpec)
@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=train_db_result,
)
def test_run_training_prophet_success(
    train_db, PredictionMediator, prophet_training_post_body
):
    "Test happy path with mocked out db"
    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Submitted",
    }

    PredictionMediator.return_value.runConfig.train_run_id = "10"
    PredictionMediator.return_value.runConfig.train_run_uuid = (
        "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"
    )
    PredictionMediator.return_value.runConfig.train_run_status = "Submitted"

    response = client.post("/training", json=prophet_training_post_body)

    # assert response.status_code == 200
    body = response.json()
    assert body == expected

    PredictionMediator.assert_called_once()
    assert (
        PredictionMediator.call_args_list[0].args[0]["models"]
        == prophet_training_post_body["models"]
    )


@patch("prediction.master_server.PredictionMediator", autospec=PredictionMediatorSpec)
@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=train_db_result,
)
def test_run_training_holtwinters_success(
    train_db, PredictionMediator, holtwinters_training_post_body
):
    "Test happy path with mocked out db"
    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Submitted",
    }

    PredictionMediator.return_value.runConfig.train_run_id = "10"
    PredictionMediator.return_value.runConfig.train_run_uuid = (
        "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"
    )
    PredictionMediator.return_value.runConfig.train_run_status = "Submitted"

    response = client.post("/training", json=holtwinters_training_post_body)

    assert response.status_code == 200
    body = response.json()
    assert body == expected

    PredictionMediator.assert_called_once()
    assert (
        PredictionMediator.call_args_list[0].args[0]["models"]
        == holtwinters_training_post_body["models"]
    )


@patch("prediction.master_server.PredictionMediator", autospec=PredictionMediatorSpec)
@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=train_db_result,
)
def test_run_training_arima_success(
    train_db, PredictionMediator, arima_training_post_body
):
    "Test happy path with mocked out db"
    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Submitted",
    }

    PredictionMediator.return_value.runConfig.train_run_id = "10"
    PredictionMediator.return_value.runConfig.train_run_uuid = (
        "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"
    )
    PredictionMediator.return_value.runConfig.train_run_status = "Submitted"

    response = client.post("/training", json=arima_training_post_body)

    assert response.status_code == 200
    body = response.json()
    assert body == expected

    PredictionMediator.assert_called_once()

    assert (
        PredictionMediator.call_args_list[0].args[0]["models"]
        == arima_training_post_body["models"]
    )


def test_run_training_arima_bad_inputs(arima_training_post_body):
    "Test validator on random var."

    arima_training_post_body["models"]["ARIMA"]["model_time_interval"] = "Fail"

    response = client.post("/training", json=arima_training_post_body)

    assert response.status_code == 422

    arima_training_post_body["models"]["ARIMA"]["model_time_interval"] = "M"
    arima_training_post_body["models"]["ARIMA"]["model_config"]["parameters"][
        "hyperparameters"
    ] = {"disp": 0}

    response = client.post("/training", json=arima_training_post_body)

    assert response.status_code == 422


@patch("prediction.master_server.PredictionMediator", autospec=PredictionMediatorSpec)
@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=train_db_result,
)
def test_run_training_arima_with_defaults(
    train_db, PredictionMediator, arima_training_post_body
):
    "Test build defaults by the validator"
    arima_training_post_body_init = copy.deepcopy(arima_training_post_body)
    del arima_training_post_body["models"]["ARIMA"]["model_config"]

    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Submitted",
    }

    PredictionMediator.return_value.runConfig.train_run_id = "10"
    PredictionMediator.return_value.runConfig.train_run_uuid = (
        "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"
    )
    PredictionMediator.return_value.runConfig.train_run_status = "Submitted"
    response = client.post("/training", json=arima_training_post_body)

    assert response.status_code == 200
    body = response.json()
    assert body == expected
    PredictionMediator.assert_called_once()

    assert PredictionMediator.call_args_list[0].args[0]["models"]["ARIMA"][
        "model_config"
    ]["parameters"]["d"] == [0, 1]

    # Overwrite the only value in the above call that is not defaulted
    PredictionMediator.call_args_list[0].args[0]["models"]["ARIMA"]["model_config"][
        "parameters"
    ]["d"] = [1]

    assert (
        PredictionMediator.call_args_list[0].args[0]["models"]
        == arima_training_post_body_init["models"]
    )


@patch("prediction.master_server.PredictionMediator", autospec=PredictionMediatorSpec)
def test_run_training_empty_json(PredictionMediator):
    "Test happy path with mocked out db"
    response = client.post("/training", json={})
    assert response.status_code == 422
    body = response.json()
    assert body == {
        "detail": [
            {
                "loc": ["body", "training"],
                "msg": "field required",
                "type": "value_error.missing",
            }
        ]
    }
    PredictionMediator.assert_not_called()


@patch("prediction.master_server.PredictionMediator", autospec=PredictionMediatorSpec)
def test_run_training_invalid_model_names(PredictionMediator):
    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Completed",
    }
    post_body = {
        "training": {
            "train_name": "model-minimum-config-postman",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-12-02",
            "model_names": ["ARIMA", "HOLTWINTERS", "GOODPROPHET"],
        },
    }
    response = client.post("/training", json=post_body)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"][0]["ctx"]["given"] == "GOODPROPHET"
    assert body["detail"][0]["ctx"]["permitted"] == PREDICTION_CONTEXT.supported_models
    PredictionMediator.assert_not_called()


@patch("prediction.master_server.PredictionMediator", autospec=PredictionMediatorSpec)
@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=train_db_result,
)
def test_run_training_arima_with_features_success(
    train_db, PredictionMediator, exog_training_post_body
):
    "Test happy path with mocked out db"
    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Submitted",
    }

    PredictionMediator.return_value.runConfig.train_run_id = "10"
    PredictionMediator.return_value.runConfig.train_run_uuid = (
        "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"
    )
    PredictionMediator.return_value.runConfig.train_run_status = "Submitted"

    response = client.post("/training", json=exog_training_post_body)

    assert response.status_code == 200
    body = response.json()
    assert body == expected

    PredictionMediator.assert_called_once()
    assert (
        PredictionMediator.call_args_list[0].args[0]["models"]
        == exog_training_post_body["models"]
    )


@patch("prediction.master_server.PredictionMediator", autospec=PredictionMediatorSpec)
def test_run_training_bad_json(PredictionMediator):
    "Test out invalid UUID"
    response = client.post("/training", data="My name is JSON")
    assert response.status_code == 422
    body = response.json()
    error = body["detail"][0]
    assert error["loc"] == ["body", 0]
    assert error["type"] == "value_error.jsondecode"
    PredictionMediator.assert_not_called()


### Test Get Prediction Results ###
@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=train_db_result,
)
def test_fetch_train_results(train_db):
    "Test happy path with mocked out db"
    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Completed",
        "resultLocation": "tests/resources/data/score/model-minimum-config-postman_2021-01-05-18.29.16.csv",
        "trainStartTs": "2021-01-05T18:29:16.246159",
        "trainEndTs": "2021-01-05T18:55:12.984947",
        "updateTs": "2021-01-05T18:55:12.984947",
    }
    response = client.get(
        "/trainings/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == expected
    train_db.assert_called_once()
    train_db.assert_called_with(UUID("c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"), "devin")


@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=None,
)
def test_fetch_train_results_invalid_run_id(train_db):
    "Test out invalid UUID"
    response = client.get("/trainings/6a32-4ba1-a385-2eb2cfbd4e11")
    assert response.status_code == 422
    body = response.json()
    assert len(body["detail"]) == 1
    error = body["detail"][0]
    assert error["loc"] == ["path", "model_training_run_id"]
    assert error["msg"] == "value is not a valid uuid"
    train_db.assert_not_called()


@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=None,
)
def test_fetch_train_results_missing_run_id(train_db):
    "Test out missing UUID"
    response = client.get(
        "/trainings/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )
    assert response.status_code == 404
    body = response.json()
    assert body == {"detail": "Train run ID not found"}
    train_db.assert_called_once()
    train_db.assert_called_with(UUID("c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"), "devin")


###############################################################
#                       Predictions                           #
###############################################################
prediction_db_result = SimpleNamespace(
    **{
        "score_run_uuid": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "score_status": "Completed",
        "score_data_location": "tests/resources/data/score/model-minimum-config-postman_2021-01-05-18.29.16.csv",
        "score_start_ts": datetime.fromisoformat("2021-01-05T18:29:16.246159"),
        "score_end_ts": datetime.fromisoformat("2021-01-05T18:55:12.984947"),
        "update_ts": datetime.fromisoformat("2021-01-05T18:55:12.984947"),
    }
)

prediction_db_result_failed = SimpleNamespace(
    **{
        "score_run_uuid": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "score_status": "Failed to write to s3",
        "score_data_location": "tests/resources/data/model-minimum-config-postman_2021-01-05-18.29.16.csv",
        "score_start_ts": datetime.fromisoformat("2021-01-05T18:29:16.246159"),
        "score_end_ts": datetime.fromisoformat("2021-01-05T18:55:12.984947"),
        "update_ts": datetime.fromisoformat("2021-01-05T18:55:12.984947"),
    }
)


### Test Generate Predictions ###
class PredictionMediatorSpec(PM):
    scoreConfig = SimpleNamespace(
        **{
            "score_run_uuid": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
            "score_run_id": "10",
            "score_status": "Submitted",
        }
    )

    def predict():
        return "Submitted"


@patch("prediction.master_server.PredictionMediator", autospec=PredictionMediatorSpec)
def test_generate_predictions(PredictionMediator):
    "Test happy path with mocked out db"
    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Submitted",
    }
    post_body = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["ARIMA"],
            "prediction_steps": 12,
            "prediction_count": 12,
        },
        "training": {
            "train_name": "model-minimum-config-postman",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
            "prediction_steps": "0",
        },
    }
    PredictionMediator.return_value.scoreConfig.score_run_id = "10"
    PredictionMediator.return_value.scoreConfig.score_run_uuid = (
        "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"
    )
    PredictionMediator.return_value.scoreConfig.score_status = "Submitted"

    response = client.post("/predictions", json=post_body)
    assert response.status_code == 200
    body = response.json()
    assert body == expected
    PredictionMediator.assert_called_once()

    params = PredictionMediator.call_args.kwargs["params"]
    assert (
        params["scoring"]["target_data_location"]
        == "tests/resources/data/test_single_target.csv"
    )


@patch(
    "prediction.master_server.pred_db.get_score_run_config_by_uuid",
    return_value=prediction_db_result,
)
def test_generate_predictions_empty_json(pred_db):
    "Test happy path with mocked out db"
    response = client.post("/predictions", json={})
    assert response.status_code == 422
    body = response.json()
    assert body == {
        "detail": [
            {
                "loc": ["body", "scoring"],
                "msg": "field required",
                "type": "value_error.missing",
            }
        ]
    }
    pred_db.assert_not_called()


@patch(
    "prediction.master_server.pred_db.get_score_run_config_by_uuid",
    return_value=prediction_db_result,
)
def test_generate_predictions_invalid_model_names(pred_db):
    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Completed",
    }
    post_body = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["GOODPROPHET"],
            "prediction_steps": 12,
            "prediction_count": 12,
        },
        "training": {
            "train_name": "model-minimum-config-postman",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-13-02",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
        },
    }
    response = client.post("/predictions", json=post_body)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"][0]["type"] == "type_error"
    assert body["detail"][0]["loc"] == ["body", "training", "validation_data_end_dtm"]
    assert body["detail"][1]["ctx"]["given"] == "GOODPROPHET"
    assert body["detail"][1]["ctx"]["permitted"] == PREDICTION_CONTEXT.supported_models
    pred_db.assert_not_called()


@patch(
    "prediction.master_server.pred_db.get_score_run_config_by_uuid",
    return_value=None,
)
def test_generate_prediction_bad_json(pred_db):
    "Test out invalid UUID"
    response = client.post("/predictions", data="My name is JSON")
    assert response.status_code == 422
    body = response.json()
    error = body["detail"][0]
    assert error["loc"] == ["body", 0]
    assert error["type"] == "value_error.jsondecode"
    pred_db.assert_not_called()


@patch("prediction.master_server.PredictionMediator", autospec=PredictionMediatorSpec)
def test_generate_predictions_with_features(PredictionMediator):
    "Test happy path with mocked out db and with external features"
    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Submitted",
    }
    post_body = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "feature_data_location": "tests/resources/data/test_single_exog.csv",
            "target_feature_mapping_location": "tests/resources/data/test_single_exog_map.csv",
            "model_names": ["ARIMA"],
            "prediction_steps": 12,
            "prediction_count": 12,
        },
        "training": {
            "train_name": "model-minimum-config-postman",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "train_data_end_dtm": "1990-01-31",
            "test_data_end_dtm": "1990-02-03",
            "validation_data_end_dtm": "1990-03-02",
            "model_names": ["ARIMA"],
        },
    }
    PredictionMediator.return_value.scoreConfig.score_run_id = "10"
    PredictionMediator.return_value.scoreConfig.score_run_uuid = (
        "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"
    )
    PredictionMediator.return_value.scoreConfig.score_status = "Submitted"

    response = client.post("/predictions", json=post_body)
    assert response.status_code == 200
    body = response.json()
    assert body == expected
    PredictionMediator.assert_called_once()

    params = PredictionMediator.call_args.kwargs["params"]
    assert (
        params["scoring"]["target_data_location"]
        == "tests/resources/data/test_single_target.csv"
    )


### Test Get Prediction Results ###
@patch(
    "prediction.master_server.pred_db.get_score_run_config_by_uuid",
    return_value=prediction_db_result,
)
def test_fetch_predictions(pred_db):
    "Test happy path with mocked out db"
    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Completed",
        "resultLocation": "tests/resources/data/score/model-minimum-config-postman_2021-01-05-18.29.16.csv",
        "scoreStartTs": "2021-01-05T18:29:16.246159",
        "scoreEndTs": "2021-01-05T18:55:12.984947",
        "updateTs": "2021-01-05T18:55:12.984947",
    }
    response = client.get(
        "/predictions/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == expected
    pred_db.assert_called_once()
    pred_db.assert_called_with(UUID("c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"), "devin")


@patch(
    "prediction.master_server.pred_db.get_score_run_config_by_uuid",
    return_value=None,
)
def test_fetch_predictions_invalid_run_id(pred_db):
    "Test out invalid UUID"
    response = client.get(
        "/predictions/6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )
    assert response.status_code == 422
    body = response.json()
    assert len(body["detail"]) == 1
    error = body["detail"][0]
    assert error["loc"] == ["path", "model_predict_id"]
    assert error["msg"] == "value is not a valid uuid"
    pred_db.assert_not_called()


@patch(
    "prediction.master_server.pred_db.get_score_run_config_by_uuid",
    return_value=None,
)
def test_fetch_predictions_missing_run_id(pred_db):
    "Test out missing UUID"
    response = client.get(
        "/predictions/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )
    assert response.status_code == 404
    body = response.json()
    assert body == {"detail": "Prediction ID not found"}
    pred_db.assert_called_once()
    pred_db.assert_called_with(UUID("c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"), "devin")


def status():
    response = client.get("/service/status")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_healthcheck_gtg():
    response = client.get("/service/healthcheck/gtg")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_redoc_status():
    response = client.get("/redoc")
    assert response.status_code == 200


def test_docs_status():
    response = client.get("/docs")
    assert response.status_code == 200


## Writing test cases to check if the Status messages are being updated correctly or not
class PredictionMediatorMockSuccess(PM):
    scoreConfig = SimpleNamespace(
        **{
            "score_run_uuid": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
            "score_run_id": "10",
            "score_status": "Created",
        }
    )

    def predict():
        return "Submitted"


@patch(
    "prediction.master_server.PredictionMediator",
    return_value=PredictionMediatorMockSuccess,
)
def test_predict_success(PredictionMediator):
    "Returnng a success status to see if the job status is coming in correctly"

    sample_post_body = {
        "scoring": {
            "score_name": "postman-test-score-prev-train",
            "target_data_location": "s3://prediction-services/data/test_single_target.csv",
            "score_data_location": "s3://kbs-analytics-dev-c3-public/fhr-public/lv-polypropylene-demand-forecast/output/Prediction/",
            "model_names": ["PROPHET"],
            "prediction_steps": 12,
            "prediction_count": 10,
        }
    }

    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Submitted",
    }

    PredictionMediator.return_value.scoreConfig.score_run_id = "10"
    PredictionMediator.return_value.scoreConfig.score_run_uuid = (
        "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"
    )
    PredictionMediator.return_value.scoreConfig.score_status = "Submitted"
    response = client.post("/predictions", json=sample_post_body)

    assert response.status_code == 200
    body = response.json()
    assert body == expected
    PredictionMediator.assert_called_once()


class PredictionMediatorMockFailedReturn(PM):
    scoreConfig = SimpleNamespace(
        **{
            "score_run_uuid": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
            "score_run_id": "10",
            "score_status": "Created",
        }
    )

    def predict():
        return "Failed"


@patch(
    "prediction.master_server.PredictionMediator",
    return_value=PredictionMediatorMockFailedReturn,
)
def test_predict_failure(PredictionMediator):
    "Testing when predict returns Failed status"

    sample_post_body = {
        "scoring": {
            "score_name": "postman-test-score-prev-train",
            "target_data_location": "s3://prediction-services/data/test_single_target.csv",
            "score_data_location": "s3://kbs-analytics-dev-c3-public/fhr-public/lv-polypropylene-demand-forecast/output/Prediction/",
            "model_names": ["PROPHET"],
            "prediction_steps": 12,
            "prediction_count": 10,
        }
    }

    expected = {
        "detail": {"runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11", "status": "Failed"}
    }

    PredictionMediator.return_value.scoreConfig.score_run_id = "10"
    PredictionMediator.return_value.scoreConfig.score_run_uuid = (
        "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11"
    )
    PredictionMediator.return_value.scoreConfig.score_status = "Submitted"
    response = client.post("/predictions", json=sample_post_body)

    assert response.status_code == 400
    body = response.json()
    assert body == expected
    PredictionMediator.assert_called_once()


##-----------------------Training Deletion----------------------

mock_train_db_result = SimpleNamespace(
    **{
        "train_run_uuid": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "train_status": "Completed",
        "train_data_location": "tests/resources/data/score/model-minimum-config-postman_2021-01-05-18.29.16.csv",
        "score_data_location": "tests/resources/data/score/model-minimum-config-postman_2021-01-05-18.29.16.csv",
        "train_start_ts": datetime.fromisoformat("2021-01-05T18:29:16.246159"),
        "train_end_ts": datetime.fromisoformat("2021-01-05T18:55:12.984947"),
        "update_ts": datetime.fromisoformat("2021-01-05T18:55:12.984947"),
        "train_run_id": 15,
    }
)


mock_train_model_db_result = [
    # {
    #     "train_run_model_id": "367",
    #     "train_run_id": "263",
    #     "model_name":"ARIMA",
    #     "model_job_name":"ARIMA",
    #     "train_update_ts": datetime.fromisoformat("2021-01-05T18:29:16.246159"),
    # },
    # {
    #     "train_run_model_id": "368",
    #     "train_run_id": "263",
    #     "model_name":"HOLTWINTERS",
    #     "model_job_name":NONE,
    #     "train_update_ts": datetime.fromisoformat("2021-01-05T18:29:16.246159"),
    # },
    SimpleNamespace(
        **{
            "train_run_model_id": "369",
            "train_run_id": "263",
            "model_name": "DEEPAR",
            "model_job_name": "DEEPAR-202206010709",
            "train_update_ts": datetime.fromisoformat("2021-01-05T18:29:16.246159"),
        }
    )
]


@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=None,
)
def test_for_runconfig_not_found(MockTrainJobHandlerStatus):
    """Test for when no TrainRunConfig is found"""
    response = client.delete(
        "/trainings/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )

    expected = {"detail": "Train run ID not found"}
    assert response.status_code == 404
    body = response.json()
    assert body == expected
    MockTrainJobHandlerStatus.assert_called_once()


@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=mock_train_db_result,
)
@patch(
    "prediction.master_server.PredictionJobHandler.status_of_job",
    return_value="Untraceable",
)
def test_for_job_not_found(MockTrainJobHandlerStatus, MockTrainResults):
    """Testing for when the status of the Job is not found"""
    response = client.delete(
        "/trainings/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )

    expected = {
        "detail": {
            "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
            "status": "Could not get the status of the job",
        }
    }
    assert response.status_code == 404
    body = response.json()
    assert body == expected
    MockTrainJobHandlerStatus.assert_called_once()
    MockTrainResults.assert_called_once()


@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=mock_train_db_result,
)
@patch(
    "prediction.master_server.pred_db.get_model_configs_by_train_id",
    return_value=mock_train_model_db_result,
)
@patch(
    "prediction.master_server.SageMakerTrainer.stop_running_process",
    return_value="UNTRACEABLE",
)
def test_for_Sagemaker_job_not_found(
    MockDeepARStatus, MockModelConfig, MockTrainResults
):
    """Testing for when the status of the Sagemaker Job is not found"""
    response = client.delete(
        "/trainings/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )

    expected = {
        "detail": {
            "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
            "status": "Could not get the status of the job",
        }
    }
    assert response.status_code == 404
    body = response.json()
    assert body == expected

    MockTrainResults.assert_called_once()
    MockModelConfig.assert_called_once()
    MockDeepARStatus.assert_called_once()


@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=mock_train_db_result,
)
@patch(
    "prediction.master_server.PredictionJobHandler.status_of_job",
    return_value="Active",
)
@patch(
    "prediction.master_server.PredictionJobHandler.delete_active_job",
    return_value="Failed",
)
def test_for_job_killing_failed(
    MockDeleteJobStatus, MockTrainJobHandlerStatus, MockTrainResults
):
    """Testing for Failure"""
    response = client.delete(
        "/trainings/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )

    expected = {
        "detail": {
            "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
            "status": "Could not kill the job",
        }
    }
    assert response.status_code == 404
    body = response.json()
    assert body == expected
    MockDeleteJobStatus.assert_called_once()
    MockTrainJobHandlerStatus.assert_called_once()
    MockTrainResults.assert_called_once()


@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=mock_train_db_result,
)
@patch(
    "prediction.master_server.pred_db.get_model_configs_by_train_id",
    return_value=mock_train_model_db_result,
)
@patch(
    "prediction.master_server.SageMakerTrainer.stop_running_process",
    return_value="STOPPED",
)
def test_for_Sagemaker_job_killing_failed(
    MockDeleteJobStatus, MockModelConfig, MockTrainResults
):
    """Testing for Sagemaker killing Failure"""
    response = client.delete(
        "/trainings/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )

    expected = {
        "detail": {
            "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
            "status": "The job is currently not running.",
        }
    }
    assert response.status_code == 404
    body = response.json()
    assert body == expected

    MockDeleteJobStatus.assert_called_once()
    MockModelConfig.assert_called_once()
    MockTrainResults.assert_called_once()


@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=mock_train_db_result,
)
@patch(
    "prediction.master_server.PredictionJobHandler.status_of_job",
    return_value="Active",
)
@patch(
    "prediction.master_server.PredictionJobHandler.delete_active_job",
    return_value="Success",
)
def test_for_job_killed(
    MockDeleteJobStatus, MockTrainJobHandlerStatus, MockTrainResults
):
    """Testing for Successfull Killing of Job"""
    response = client.delete(
        "/trainings/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )

    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Killed the job successfully",
    }
    assert response.status_code == 200
    body = response.json()
    assert body == expected
    MockDeleteJobStatus.assert_called_once()
    MockTrainJobHandlerStatus.assert_called_once()
    MockTrainResults.assert_called_once()


@patch(
    "prediction.master_server.pred_db.get_train_run_config_by_uuid",
    return_value=mock_train_db_result,
)
@patch(
    "prediction.master_server.pred_db.get_model_configs_by_train_id",
    return_value=mock_train_model_db_result,
)
@patch(
    "prediction.master_server.SageMakerTrainer.stop_running_process",
    return_value="SUCCESS",
)
def test_for_sagemaker_job_killed(MockDeepARStatus, MockModelConfig, MockTrainResults):
    """Testing for Successfull Killing of Sagemaker Job"""
    response = client.delete(
        "/trainings/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )

    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Killed the job successfully",
    }
    assert response.status_code == 200
    body = response.json()
    assert body == expected
    MockDeepARStatus.assert_called_once()
    MockModelConfig.assert_called_once()
    MockTrainResults.assert_called_once()


##-----------------------Prediction Deletion----------------------
mock_prediction_db_result = SimpleNamespace(
    **{
        "score_run_uuid": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "train_run_id": "263",
        "score_status": "Completed",
        "score_data_location": "tests/resources/data/score/model-minimum-config-postman_2021-01-05-18.29.16.csv",
        "score_start_ts": datetime.fromisoformat("2021-01-05T18:29:16.246159"),
        "score_end_ts": datetime.fromisoformat("2021-01-05T18:55:12.984947"),
        "update_ts": datetime.fromisoformat("2021-01-05T18:55:12.984947"),
        "score_run_id": 20,
    }
)


@patch(
    "prediction.master_server.pred_db.get_score_run_config_by_uuid",
    return_value=None,
)
def test_for_predictionconfig_not_found(MockPredictionJobHandlerStatus):
    """Test for when no TrainRunConfig is found"""
    response = client.delete(
        "/predictions/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )

    expected = {"detail": "Prediction run ID not found"}
    assert response.status_code == 404
    body = response.json()
    assert body == expected
    MockPredictionJobHandlerStatus.assert_called_once()


@patch(
    "prediction.master_server.pred_db.get_score_run_config_by_uuid",
    return_value=mock_prediction_db_result,
)
@patch(
    "prediction.master_server.pred_db.get_model_configs_by_train_id",
    return_value=mock_train_model_db_result,
)
@patch(
    "prediction.master_server.PredictionJobHandler.status_of_job",
    return_value="Untraceable",
)
def test_for_prediction_job_not_found(
    MockPredictionJobHandlerStatus, MockModelConfig, MockPredictionResults
):
    """Testing for when the status of the Job is not found"""
    response = client.delete(
        "/predictions/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )

    expected = {
        "detail": {
            "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
            "status": "Could not get the status of the job",
        }
    }
    assert response.status_code == 404
    body = response.json()
    assert body == expected
    MockPredictionJobHandlerStatus.assert_called_once()
    MockModelConfig.assert_called_once()
    MockPredictionResults.assert_called_once()


@patch(
    "prediction.master_server.pred_db.get_score_run_config_by_uuid",
    return_value=mock_prediction_db_result,
)
@patch(
    "prediction.master_server.PredictionJobHandler.status_of_job",
    return_value="Active",
)
@patch(
    "prediction.master_server.PredictionJobHandler.delete_active_job",
    return_value="Failed",
)
def test_for_prediction_job_killing_failed(
    MockDeleteJobStatus, MockPredictionJobHandlerStatus, MockPredictionResults
):
    """Testing for Failure"""
    response = client.delete(
        "/predictions/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )

    expected = {
        "detail": {
            "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
            "status": "Could not kill the job",
        }
    }
    assert response.status_code == 404
    body = response.json()
    assert body == expected
    MockDeleteJobStatus.assert_called_once()
    MockPredictionJobHandlerStatus.assert_called_once()
    MockPredictionResults.assert_called_once()


@patch(
    "prediction.master_server.pred_db.get_score_run_config_by_uuid",
    return_value=mock_prediction_db_result,
)
@patch(
    "prediction.master_server.PredictionJobHandler.status_of_job",
    return_value="Active",
)
@patch(
    "prediction.master_server.PredictionJobHandler.delete_active_job",
    return_value="Success",
)
def test_for_prediction_job_killed(
    MockDeleteJobStatus, MockPredictionJobHandlerStatus, MockPredictionResults
):
    """Testing for Successfull Killing of Job"""
    response = client.delete(
        "/predictions/c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        headers={"X-Consumer-Username": "devin"},
    )

    expected = {
        "runId": "c452ab8e-6a32-4ba1-a385-2eb2cfbd4e11",
        "status": "Killed the job successfully",
    }
    assert response.status_code == 200
    body = response.json()
    assert body == expected
    MockDeleteJobStatus.assert_called_once()
    MockPredictionJobHandlerStatus.assert_called_once()
    MockPredictionResults.assert_called_once()
