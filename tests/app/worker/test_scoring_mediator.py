import logging
import multiprocessing as mp
import time
from datetime import datetime

import pytest
import ray
from pony.orm import *

import prediction.app.worker.scoring_mediator as score_med
import prediction.src.db.prediction_database as pred_db
from prediction.app.prediction_commands import DataCommand, ScoringCommand
from prediction.app.worker.training_handlers import TSAModeler

logger = logging.getLogger("prediction_services")

S3_PRED_PATH = "tests/resources/data/"
S3_DATA_PATH = S3_PRED_PATH + "test_single_target.csv"


@pytest.fixture
def tsaSingleTargetConfig():
    scoringDict = {
        "scoring": {
            "score_name": "score-tsa-single-target",
            "target_data_location": S3_PRED_PATH + "h0500hn_ft-worth.csv",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
            "prediction_steps": 12,
            "prediction_count": 10,
            "username": "devin",
        }
    }
    return scoringDict


@pytest.fixture
def tsaMultipleTargetsConfig():
    return {
        "scoring": {
            "score_name": "score-tsa-multiple-targets",
            "target_data_location": f"{S3_PRED_PATH}TrainingInputData_Transformed_Test_Sample_2.csv",
            "model_names": ["HOLTWINTERS", "ARIMA", "Prophet"],
            "prediction_steps": 12,
            "prediction_count": 10,
        }
    }


@pytest.fixture
def dataCommand():
    return DataCommand(
        target_data_location=S3_DATA_PATH,
        feature_data_location="",
        train_data_end_dtm=datetime.now(),
        test_data_end_dtm=datetime.now(),
        validation_data_end_dtm=datetime.now(),
        model_name="ARIMA",
    )


@pytest.fixture
def scoreArimaConfig():
    return {
        "parameters": {"trend": "add", "damped": True, "seasonal": None},
        "hyperparameters": {"optimized": True, "use_boxcox": True, "remove_bias": True},
    }


@pytest.fixture
def arimaCmd(scoreArimaConfig):
    return ScoringCommand(
        score_name="Testing score command",
        target_column="target1",
        model_name="HOLTWINTERS",
        model_config=scoreArimaConfig,
        score_status="Created",
        prediction_steps=3,
        score_time_frequency="D",
    )


@pytest.mark.skip
def testScoringMediatorCreate(seed_data, dataCommand, tempdir):
    scoreId = 15
    scoreConfig = pred_db.get_score_run_config(scoreId)
    assert scoreConfig is not None
    scoreList = pred_db.get_score_model_summaries_by_score_id(scoreId)
    assert scoreList is not None
    assert len(scoreList) > 0
    mediator = score_med.ScoringMediator(scoreConfig, scoreList)
    assert mediator is not None
    assert mediator.dataCommand is not None
    assert mediator.MODEL is not None
    assert len(mediator.MODEL) == len(scoreList)
    assert mediator.DATA is not None
    assert len(mediator.DATA) > 0


@pytest.mark.skip
def testScoringMediatorArima(seed_data, dataCommand):
    scoreId = 13
    scoreConfig = pred_db.get_score_run_config(scoreId)
    assert scoreConfig is not None
    scoreList = pred_db.get_score_model_summaries_by_score_id(scoreId)
    assert scoreList is not None
    assert len(scoreList) > 0
    mediator = score_med.ScoringMediator(scoreConfig, scoreList)
    assert mediator is not None
    assert mediator.dataCommand is not None
    assert mediator.MODEL is not None
    assert mediator.DATA is not None
    assert len(mediator.DATA) > 0
    arimaDict = mediator.score_in_parallel()
    logger.info(f"ARIMA forecast: {arimaDict}")
    assert arimaDict is not None
    assert len(arimaDict) >= 1


@pytest.mark.skip
def testScoringMediatorHoltWinterMultipleTS():
    scoringDict = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": f"{S3_PRED_PATH}TrainingInputData_Transformed_Test_Sample_2.csv",
            "model_names": ["HOLTWINTERS"],
            "prediction_steps": 12,
            "prediction_count": 10
            ##"train_run_id": "e5ff99ac-e260-49d0-934f-46c46d31f136"
        }
    }
    scoreConfig = pred_db.create_score_run_config(scoringDict)
    assert scoreConfig is not None
    scoreConfig.username = "devin"
    scoreModelList = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreModelList is not None
    assert len(scoreModelList) > 0

    mediator = score_med.ScoringMediator(scoreConfig, scoreModelList)
    assert mediator is not None
    assert mediator.dataCommand is not None
    assert mediator.MODEL is not None
    assert len(mediator.MODEL) > 1
    assert mediator.dataCommand is not None
    assert mediator.DATA is not None

    status = mediator.score_in_parallel()
    logger.info(f"*******Scoring status: {status}")
    logger.info(f"****HoltwinterMultipl Number of data columns: {len(mediator.DATA)}")
    assert len(mediator.DATA) > 0
    assert status is not None
    scoreResults = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreResults is not None
    assert len(scoreResults) > 0
    scoreSummary = scoreResults[0]
    assert scoreSummary is not None
    assert scoreSummary.forecasted_quantity is not None
    assert len(scoreSummary.forecasted_quantity) > 0

    assert len(mediator.DATA) == 1


@pytest.mark.skip
def testScoringMediatorHoltWinter(tempdir):
    scoringDict = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "score_data_location": f"{tempdir}/scores/",
            "model_names": ["HOLTWINTERS"],
            "prediction_steps": 12,
            "prediction_count": 10,
            ##"train_run_id": "e5ff99ac-e260-49d0-934f-46c46d31f136"
            "username": "devin",
        }
    }
    scoreConfig = pred_db.create_score_run_config(scoringDict)
    assert scoreConfig is not None
    scoreModelList = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreModelList is not None
    assert len(scoreModelList) > 0

    mediator = score_med.ScoringMediator(scoreConfig, scoreModelList)
    assert mediator is not None
    assert mediator.dataCommand is not None
    assert mediator.MODEL is not None
    assert mediator.DATA is not None

    status = mediator.score_in_parallel()
    logger.info(f"Scoring status: {status}")
    assert len(mediator.DATA) > 0
    assert status is not None
    scoreResults = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreResults is not None
    assert len(scoreResults) > 0
    scoreSummary = scoreResults[0]
    assert scoreSummary is not None
    assert scoreSummary.forecasted_quantity is not None
    assert len(scoreSummary.forecasted_quantity) > 0


@pytest.mark.skip
def testScoringMediatorArimaPredCount():
    scoringDict = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["ARIMA"],
            "prediction_steps": 12,
            "prediction_count": 10,
            ##"train_run_id": "e5ff99ac-e260-49d0-934f-46c46d31f136"
            "username": "devin",
        }
    }
    scoreConfig = pred_db.create_score_run_config(scoringDict)
    assert scoreConfig is not None
    scoreModelList = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreModelList is not None
    assert len(scoreModelList) > 0

    mediator = score_med.ScoringMediator(scoreConfig, scoreModelList)
    assert mediator is not None
    assert mediator.dataCommand is not None
    assert mediator.MODEL is not None
    assert mediator.DATA is not None

    status = mediator.score_in_parallel()
    logger.info(f"Scoring status: {status}")
    assert len(mediator.DATA) > 0
    assert status is not None
    scoreResults = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreResults is not None
    assert len(scoreResults) > 0
    scoreSummary = scoreResults[0]
    assert scoreSummary is not None
    assert scoreSummary.forecasted_quantity is not None
    assert len(scoreSummary.forecasted_quantity) > 0

    ##@pytest.mark.skip


@pytest.mark.skip
def testScoringMediatorProphet():
    scoringDict = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["PROPHET"],
            "prediction_steps": 12,
            "prediction_count": 10,
            ##"train_run_id": "e5ff99ac-e260-49d0-934f-46c46d31f136",
            "username": "devin",
        }
    }
    scoreConfig = pred_db.create_score_run_config(scoringDict)
    assert scoreConfig is not None
    scoreModelList = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreModelList is not None
    assert len(scoreModelList) > 0

    mediator = score_med.ScoringMediator(scoreConfig, scoreModelList)
    assert mediator is not None
    assert mediator.dataCommand is not None
    assert mediator.MODEL is not None
    assert mediator.DATA is not None

    status = mediator.score_in_parallel()
    logger.info(f"Scoring status: {status}")
    assert len(mediator.DATA) > 0
    assert status is not None
    scoreResults = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreResults is not None
    assert len(scoreResults) > 0
    scoreSummary = scoreResults[0]
    assert scoreSummary is not None
    assert scoreSummary.forecasted_quantity is not None
    assert len(scoreSummary.forecasted_quantity) > 0


@pytest.mark.skip(reason="WIP in scoring flow.")
def test_scoring_mediator_deepar():
    """ Test Scoring Mediator for DeepAR """

    scoring_dict = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["DEEPAR"],
            "prediction_steps": 4,
            "prediction_count": 3,
            "username": "neelesh",
        }
    }
    score_config = pred_db.create_score_run_config(scoring_dict)
    assert score_config is not None
    score_model_list = pred_db.get_score_model_summaries_by_score_id(
        score_config.score_run_id
    )
    assert score_model_list is not None
    assert len(score_model_list) > 0

    mediator = score_med.ScoringMediator(score_config, score_model_list)
    assert mediator is not None
    assert mediator.dataCommand is not None
    assert mediator.MODEL is not None
    assert mediator.DATA is not None

    status = mediator.score_in_parallel()
    logger.info(f"Scoring status: {status}")
    assert len(mediator.DATA) > 0
    assert status is not None
    score_results = pred_db.get_score_model_summaries_by_score_id(
        score_config.score_run_id
    )
    assert score_results is not None
    assert len(score_results) > 0
    scoreSummary = score_results[0]
    assert scoreSummary is not None
    assert scoreSummary.forecasted_quantity is not None
    assert len(scoreSummary.forecasted_quantity) > 0


@pytest.mark.skip
def testScoringMediatorTSA():
    scoringDict = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
            "prediction_steps": 12,
            "prediction_count": 10,
            ##"train_run_id": "e5ff99ac-e260-49d0-934f-46c46d31f136",
            "username": "devin",
        }
    }
    scoreConfig = pred_db.create_score_run_config(scoringDict)
    assert scoreConfig is not None
    scoreModelList = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreModelList is not None
    assert len(scoreModelList) > 0

    mediator = score_med.ScoringMediator(scoreConfig, scoreModelList)
    assert mediator is not None
    assert mediator.dataCommand is not None
    assert mediator.MODEL is not None
    assert mediator.DATA is not None

    status = mediator.score_in_parallel()
    logger.info(f"Scoring status: {status}")
    assert len(mediator.DATA) > 0
    assert status is not None
    scoreResults = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreResults is not None
    assert len(scoreResults) > 0
    scoreSummary = scoreResults[0]
    assert scoreSummary is not None
    assert scoreSummary.forecasted_quantity is not None
    assert len(scoreSummary.forecasted_quantity) > 0


@pytest.mark.skip
def test_scoring_mediator_deepar_modeler():
    """ Test Scoring Mediator DeepAR MODELER """
    # TODO: WIP
    scoring_dict = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["DEEPAR"],
            "prediction_steps": 4,
            "prediction_count": 3,
            "username": "neelesh",
        }
    }
    score_config = pred_db.create_score_run_config(scoring_dict)
    assert score_config is not None
    score_model_list = pred_db.get_score_model_summaries_by_score_id(
        score_config.score_run_id
    )
    assert score_model_list is not None
    assert len(score_model_list) > 0

    mediator = score_med.ScoringMediator(score_config, score_model_list)
    assert mediator is not None
    assert mediator.dataCommand is not None
    assert mediator.MODEL is not None
    assert mediator.DATA is not None

    status = mediator.score_in_parallel()
    logger.info(f"Scoring status: {status}")
    assert len(mediator.DATA) > 0
    assert status is not None
    score_results = pred_db.get_score_model_summaries_by_score_id(
        score_config.score_run_id
    )
    assert score_results is not None
    assert len(score_results) > 0
    score_summary = score_results[0]
    assert score_summary is not None
    assert score_summary.forecasted_quantity is not None
    assert len(score_summary.forecasted_quantity) > 0


@pytest.mark.skip
def testScoringMediatorTSAUsingPrev():
    ## Create database record for training
    trainConfigDict = {
        "training": {
            "train_name": "test-scoring-mediator-prev",
            "target_data_location": "tests/resources/data/h0500hn_ft-worth.csv",
            "train_data_end_dtm": "2019-12-01",
            "test_data_end_dtm": "2020-06-01",
            "validation_data_end_dtm": "2020-11-01",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
        }
    }
    config = pred_db.create_train_run_config(trainConfigDict, username="devin")
    assert config is not None
    trainRunUuid = config.train_run_uuid
    assert trainRunUuid is not None

    ## ScoreRunConfig--------------------------------------
    scoringDict = {
        "scoring": {
            "score_name": "score-minimum-config-manager",
            "target_data_location": "tests/resources/data/test_single_target.csv",
            "model_names": ["ARIMA", "HOLTWINTERS", "PROPHET"],
            "prediction_steps": 12,
            "prediction_count": 10,
            "train_run_id": trainRunUuid,
            "username": "devin",
        }
    }
    scoreConfig = pred_db.create_score_run_config(scoringDict)
    assert scoreConfig is not None
    scoreModelList = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreModelList is not None
    assert len(scoreModelList) > 0

    mediator = score_med.ScoringMediator(scoreConfig, scoreModelList)
    assert mediator is not None
    assert mediator.dataCommand is not None
    assert mediator.MODEL is not None
    assert mediator.DATA is not None

    status = mediator.score_in_parallel()
    logger.info(f"Scoring status: {status}")
    assert len(mediator.DATA) > 0
    assert status is not None
    scoreResults = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreResults is not None
    assert len(scoreResults) > 0
    scoreSummary = scoreResults[0]
    assert scoreSummary is not None
    assert scoreSummary.forecasted_quantity is not None
    assert len(scoreSummary.forecasted_quantity) > 0


##-----------------------RAY TRAIN PARALLEL------------------
@pytest.mark.skip
def testParallelScoreSingleTarget(tsaSingleTargetConfig):
    scoreConfig = pred_db.create_score_run_config(tsaSingleTargetConfig)
    assert scoreConfig is not None
    scoreModelList = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreModelList is not None
    assert len(scoreModelList) > 0
    mediator = score_med.ScoringMediator(scoreConfig, scoreModelList)
    assert mediator is not None
    assert mediator.dataCommand is not None
    assert mediator.MODEL is not None

    mediator._prepare_score_data()
    assert mediator.DATA is not None
    assert len(mediator.DATA) > 0

    scorer = score_med.ParallelScore()
    assert scorer is not None
    assert scorer.status is None
    for currData in mediator.DATA:
        df = currData["data"]
        assert df is not None
        col = currData["column"]
        assert col is not None
        logger.info(f"START - Scoring for data: {col}")
        handler = TSAModeler(target_references_id=1)
        assert handler is not None
        scorer.score_single_target(handler, mediator.MODEL, col)
        assert scorer.status is not None


@pytest.mark.skip
def testScoringMediatorParallelScore(tsaSingleTargetConfig):
    scoreConfig = pred_db.create_score_run_config(tsaSingleTargetConfig)
    assert scoreConfig is not None
    scoreModelList = pred_db.get_score_model_summaries_by_score_id(
        scoreConfig.score_run_id
    )
    assert scoreModelList is not None
    assert len(scoreModelList) > 0
    mediator = score_med.ScoringMediator(scoreConfig, scoreModelList)
    assert mediator is not None
    assert mediator.dataCommand is not None
    assert mediator.MODEL is not None

    ## Testing PARALLEL scoring
    ray.init(local_mode=True, num_cpus=mp.cpu_count())
    tic = time.perf_counter()
    result = mediator.score_in_parallel()
    toc = time.perf_counter()
    logger.info(
        f"*********Result of SCORING: {result}, time elapsed: {toc-tic:0.4f} seconds"
    )

    assert mediator.DATA is not None
    assert len(mediator.DATA) > 0
    assert result is not None
    assert result == "Completed"
    ray.shutdown()
