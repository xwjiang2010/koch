import logging
from typing import Dict

import pytest

from prediction.app.prediction_commands import (
    Command,
    HyperparameterTuning,
    RayDistributed,
    ScoringCommand,
    TrainingModel,
)
from prediction.src.static import PREDICTION_CONTEXT

logger = logging.getLogger(PREDICTION_CONTEXT.logger_name)


def testCommand():
    cmd = Command()
    assert cmd.evt_id is not None
    assert cmd.evt_create_ts is not None


def testTrainingModel():
    trainName = "TrainTest"
    trainDesc = "Training test command"
    trainJobType = "Sequential"
    trainCmd = TrainingModel(
        train_model_id=1,
        train_name=trainName,
        train_description=trainDesc,
        train_job_type=trainJobType,
        model_location="s3://",
        model_name="ARIMA",
        model_config=None,
        model_metric="MAPE",
        model_time_frequency="M",
        model_params={},
        model_hyperparams={},
        modeler=None,
        data_preprocessor=None,
        metrics={},
    )
    assert trainCmd.evt_create_ts is not None
    assert trainCmd.evt_id is not None
    assert trainCmd.train_name is not None
    assert trainCmd.train_name == trainName
    assert trainCmd.train_description == trainDesc
    assert trainCmd.train_job_type == trainJobType
    assert trainCmd.metrics == {}


def testHyperparamterTuning():
    trainName = "TrainTest"
    trainDesc = "Training test command"
    trainJobType = "Sequential"
    algorithm = "GRID-SEARCH"
    trainCmd = HyperparameterTuning(
        train_model_id=1,
        train_name=trainName,
        train_description=trainDesc,
        train_job_type=trainJobType,
        metrics={},
        model_location="s3://",
        model_name="ARIMA",
        model_config=None,
        model_metric="MAPE",
        model_time_frequency="M",
        model_params={},
        model_hyperparams={},
        hyperparam_algorithm=algorithm,
        modeler=None,
        data_preprocessor=None,
    )
    assert trainCmd.evt_create_ts is not None
    assert trainCmd.evt_id is not None
    assert trainCmd.train_name is not None
    assert trainCmd.train_name == trainName
    assert trainCmd.train_description == trainDesc
    assert trainCmd.train_job_type == trainJobType
    assert trainCmd.hyperparam_algorithm == algorithm
    assert trainCmd.metrics == {}
    arimaParams = trainCmd.model_params
    assert arimaParams is not None


@pytest.mark.skip
def testRayCommandParallel():
    runId = 10
    rayCmd = RayDistributed(
        run_id=runId,
        entry_point=None,
    )
    assert rayCmd.evt_id is not None
    assert rayCmd.evt_create_ts is not None
    assert rayCmd.run_id is not None
    assert rayCmd.cluster_definition is not None
    assert isinstance(rayCmd.cluster_definition, Dict) is True
    name = rayCmd.cluster_definition["metadata"]["name"]
    assert name is not None


@pytest.mark.skip
def testRayDistributedDefaults():
    runId = 12
    rayCmd = RayDistributed(
        run_id=runId,
        ##cluster_namespace_JSON=kube_dist.clusterDEPLOYMENTJSON,
        ##deployment_file_header=kube_dist.headDEPLOYMENTJSON,
        ##deployement_cluster=kube_dist.raySERVICEJSON,
        ##cluster_definition=kube_dist.driverJOBJSON,
        ##deployment_file_worker=kube_dist.workerDEPLOYMENTJSON,
        entry_point=None,
    )

    assert rayCmd.run_id is not None
    assert rayCmd.run_id == runId
    assert rayCmd.cluster_definition is not None
    assert isinstance(rayCmd.cluster_definition, Dict) is True
    name = rayCmd.cluster_definition["metadata"]["namespace"]
    assert name is not None
    assert len(name) > 0
    args = rayCmd.job_definition["spec"]["template"]["spec"]["containers"][0]["args"]
    assert args is not None
    assert len(args) > 0
    assert rayCmd.evt_id is not None
    assert rayCmd.evt_create_ts is not None
    logger.debug(f"Ray Distributed Kube: {rayCmd.job_definition}")
    spec = rayCmd.job_definition["spec"]
    ##assert rayCmd.deployement_cluster['spec']['replicas'] == 1
    assert "replicas" not in spec
    assert rayCmd.job_definition["spec"]["template"]["spec"]["restartPolicy"] == "Never"


def testScoringCommand():
    scoreModelConfigs = {
        "parameters": {"p": 1, "d": 1, "q": 0},
        "hyperparameters": {"disp": 0},
    }

    scoreCmd = ScoringCommand(
        score_name="Testing score command",
        target_column="t1",
        model_name="ARIMA",
        model_config=scoreModelConfigs,
        score_status="Created",
        prediction_steps=3,
        score_time_frequency="D",
        model_location=PREDICTION_CONTEXT.S3_PRED_MODEL_PATH,
    )
    assert scoreCmd.score_name == "Testing score command"
    assert scoreCmd.score_time_frequency == "D"
    assert scoreCmd.target_column == "t1"
    assert scoreCmd.model_config is not None
    assert scoreCmd.prediction_steps == 3
    assert scoreCmd.score_loss_function == "MAPE"
    ##assert scoreCmd.prediction_count == 3
    assert scoreCmd.model_config is not None
    assert scoreCmd.model_config == scoreModelConfigs
    assert scoreCmd.model_params is not None
    assert scoreCmd.model_params == scoreModelConfigs["parameters"]
    assert scoreCmd.model_hyperparams is not None
    assert scoreCmd.model_hyperparams == scoreModelConfigs["hyperparameters"]
    assert scoreCmd.model_location == PREDICTION_CONTEXT.S3_PRED_MODEL_PATH


def testScoringCommand2():
    scoreModelConfigs = {
        "parameters": {"p": 1, "d": 1, "q": 0},
        "hyperparameters": {"disp": 0},
    }

    scoreCmd = ScoringCommand(
        score_name="Testing score command",
        target_column="t1",
        model_name="ARIMA",
        model_config=scoreModelConfigs,
        score_status="Created",
        prediction_steps=3,
        score_time_frequency="D",
        model_location=PREDICTION_CONTEXT.S3_PRED_MODEL_PATH,
    )
    assert scoreCmd.score_name == "Testing score command"
    assert scoreCmd.score_time_frequency == "D"
    assert scoreCmd.target_column == "t1"
    assert scoreCmd.model_config is not None
    assert scoreCmd.prediction_steps == 3
    assert scoreCmd.score_loss_function == "MAPE"
    ##assert scoreCmd.prediction_count == 3
    assert scoreCmd.model_config is not None
    assert scoreCmd.model_config == scoreModelConfigs
    assert scoreCmd.model_params is not None
    assert scoreCmd.model_params == scoreModelConfigs["parameters"]
    assert scoreCmd.model_hyperparams is not None
    assert scoreCmd.model_hyperparams == scoreModelConfigs["hyperparameters"]
    assert scoreCmd.model_location == PREDICTION_CONTEXT.S3_PRED_MODEL_PATH
