import uuid
from datetime import datetime
from typing import List

import simplejson
from loguru import logger
from pony.orm import (
    Database,
    Json,
    Optional,
    PrimaryKey,
    Required,
    Set,
    commit,
    db_session,
    select,
)
from pony.orm.core import desc

from prediction.app.prediction_commands import TrainModelTrial
from prediction.src.static import PREDICTION_CONTEXT

db = Database()


class TrainRunConfig(db.Entity):
    _table_ = ("kgsa-core-prediction-v1", "train_run_config")
    train_run_id = PrimaryKey(int, auto=True)
    train_run_uuid = Required(uuid.UUID)  # , default = uuid.uuid4)
    train_exec_environment = Required(str)
    train_data_storage_type = Optional(str)
    train_data_file_type = Required(str)
    train_name = Required(str)
    train_description = Optional(str)
    train_task_type = Required(str)
    train_job_type = Required(str)
    model_type = Optional(str)
    target_data_location = Required(str)
    feature_data_location = Optional(str)
    target_feature_mapping_location = Optional(str)
    score_data_location = Optional(str)
    feature_columns = Optional(Json)
    model_version = Optional(str)
    train_data_end_dtm = Required(datetime)
    test_data_end_dtm = Required(datetime)
    validation_data_end_dtm = Required(datetime)
    model_artifacts_location = Optional(str)
    data_version = Optional(str)
    train_status = Required(str)
    train_start_ts = Optional(datetime)
    train_end_ts = Optional(datetime)
    jobs_count = Optional(int)
    update_ts = Required(
        datetime, sql_default="CURRENT_TIMESTAMP"
    )  # Required(datetime, auto=True)
    models = Set("TrainRunModelConfig")
    username = Required(str)
    prediction_steps = Optional(int)


class TrainRunModelConfig(db.Entity):
    _table_ = ("kgsa-core-prediction-v1", "train_run_model_config")
    train_run_model_id = PrimaryKey(int, auto=True)
    train_run_id = Required(TrainRunConfig)
    model_name = Required(str)
    model_config = Required(Json)
    model_loss_function = Optional(str)
    model_time_interval = Required(str)
    model_artifacts_location = Optional(str)
    model_job_name = Optional(str)
    tune_algorithm = Optional(str)
    train_start_ts = Optional(datetime)
    train_end_ts = Optional(datetime)
    update_ts = Required(
        datetime, sql_default="CURRENT_TIMESTAMP"
    )  # Required(datetime, auto=True)
    trials = Set("TrainRunTrial")
    metrics = Optional(Json)


class TrainRunTrial(db.Entity):
    _table_ = ("kgsa-core-prediction-v1", "train_run_trial")
    train_run_trial_id = PrimaryKey(int, auto=True)
    train_run_model_id = Required(TrainRunModelConfig)
    target_column_name = Required(str)
    feature_columns = Optional(Json)
    train_model_config = Required(Json)
    model_loss_function = Optional(str)
    train_score = Optional(float)
    test_score = Optional(float)
    validation_score = Optional(float)
    trial_status = Optional(str)
    # trial_resources = Optional(str)
    model_trial_location = Optional(str)
    trial_start_ts = Optional(datetime)
    trial_end_ts = Optional(datetime)
    update_ts = Required(datetime, auto=True)
    metrics = Optional(Json)


class ScoreRunConfig(db.Entity):
    _table_ = ("kgsa-core-prediction-v1", "score_run_config")
    score_run_id = PrimaryKey(int, auto=True)
    score_run_uuid = Required(uuid.UUID)
    score_exec_environment = Required(str)
    score_data_storage_type = Optional(str)
    score_data_file_type = Required(str)
    score_name = Required(str)
    score_description = Optional(str)
    train_run_id = Optional(int)
    model_type = Optional(str)
    target_data_location = Required(str)
    feature_data_location = Optional(str)
    target_feature_mapping_location = Optional(str)
    score_data_location = Required(str)
    data_version = Optional(str)
    score_status = Required(str)
    score_start_ts = Optional(datetime)
    score_end_ts = Optional(datetime)
    update_ts = Required(datetime, auto=True)
    scores = Set("ScoreModelSummary")
    username = Required(str)


class ScoreModelSummary(db.Entity):
    _table_ = ("kgsa-core-prediction-v1", "score_model_summary")
    score_model_id = PrimaryKey(int, auto=True)
    score_run_id = Required(ScoreRunConfig)
    model_name = Required(str)
    model_config = Required(Json)
    score_loss_function = Optional(str)
    score_time_interval = Required(str)
    prediction_steps = Required(int)
    prediction_count = Optional(int)
    score_start_ts = Optional(datetime)
    score_end_ts = Optional(datetime)
    target_column_name = Optional(str)
    score_trial_status = Optional(str)
    forecast_ts = Optional(datetime)
    forecasted_quantity = Optional(Json)
    forecasted_interval = Optional(Json)
    actual_quantity = Optional(Json)
    update_ts = Required(datetime, auto=True)


db.bind("postgres", PREDICTION_CONTEXT.db_uri)

# NOTE :: Disabled to reduce the log volume, we may want to make this configurable
# sql_debug(True)
db.generate_mapping(create_tables=False)


@db_session
def create_score_run_config(scoreParams: dict) -> ScoreRunConfig:
    # try:
    master_params = scoreParams.get("master", {})
    params = scoreParams["scoring"]
    logger.debug(f"Database insertion of ScoreRunConfig of input: {params}")
    status = "Created"
    modelType = "BEST"
    models = params.get("model_names", [])
    if "ENSEMBLE" in models:
        models = PREDICTION_CONTEXT.ensemble_base_models
        modelType = "ENSEMBLE"
    scoreConfig = ScoreRunConfig(
        score_run_uuid=uuid.uuid4(),
        score_exec_environment=master_params.get(
            "exec_environment", PREDICTION_CONTEXT.default_execution_environment
        ),
        score_data_storage_type=master_params.get(
            "data_storage_type", PREDICTION_CONTEXT.default_data_storage_type
        ),
        score_data_file_type=master_params.get(
            "data_file_type", PREDICTION_CONTEXT.default_file_type
        ),
        score_name=params["score_name"],
        score_description=params.get("score_description", ""),
        model_type=modelType,
        target_data_location=params["target_data_location"],
        feature_data_location=params.get("feature_data_location", ""),
        target_feature_mapping_location=params.get(
            "target_feature_mapping_location", ""
        ),
        score_data_location=params.get(
            "score_data_location", PREDICTION_CONTEXT.S3_SCORE_DATA_PATH
        ),
        data_version=params.get("data_version", ""),
        score_status=status,
        username=params["username"],
    )
    trainUuid = params.get("train_run_id", None)
    trainRunId = None
    if trainUuid is not None:
        trainConfig = get_train_run_config_by_uuid(trainUuid, params["username"])
        if trainConfig is None:
            logger.info(
                f"Can not find the training data for: {trainUuid}. I expect this to be here."
            )
        if trainConfig and trainConfig.train_run_id:
            trainRunId = trainConfig.train_run_id
            scoreConfig.train_run_id = trainRunId

    commit()

    # ScoreModelSummary----------------------
    scoreRunId = scoreConfig.score_run_id
    predSteps = params["prediction_steps"]
    predCnt = params.get(
        "prediction_count", PREDICTION_CONTEXT.default_prediction_count
    )
    timeFreq = params.get("time_interval", "")
    lossFunc = params.get("loss_function", "")

    # Prediction using previously trained models
    if trainUuid is not None:
        ##trainRunId = scoreConfig.train_run_id
        logger.info(
            f"Populating ScoreModelSummary using previously trained model id: {trainUuid}"
        )
        ## For tuning find the best config:
        ## TODO: by target as well
        trainTrials = get_train_run_trials_by_train_id(trainRunId)
        trialDict = {}
        for trainRunTrial in trainTrials:
            trainModelId = trainRunTrial.train_run_model_id.train_run_model_id
            if trainModelId in trialDict:
                prevTrial = trialDict[trainModelId]
                prevValidationScore = prevTrial.validation_score
                currValidationScore = trainRunTrial.validation_score
                if (
                    not prevValidationScore
                    or prevValidationScore < 0
                    or prevValidationScore <= currValidationScore
                ):
                    trialDict[trainModelId] = trainRunTrial
            else:
                trialDict[trainModelId] = trainRunTrial

        trainModelConfigs = get_model_configs_by_train_id(trainRunId)
        for trainModelConfig in trainModelConfigs:
            logger.debug(
                f"Populating ScoreModelSummary using trained model {trainModelConfig}"
            )
            currTrainModelId = trainModelConfig.train_run_model_id

            if currTrainModelId in trialDict:
                currTrainRunTrial = trialDict[currTrainModelId]
                trialModelParams = currTrainRunTrial.train_model_config
                if (
                    "parameters" in trialModelParams
                    and "hyperparameters" in trialModelParams
                ):
                    scoreModelConfig = trialModelParams
                else:
                    if "parameters" in trialModelParams:
                        trialModelParams = trialModelParams["parameters"]

                    trainRunModelConfig = trainModelConfig.model_config
                    scoreModelConfig = {
                        "parameters": trialModelParams,
                        "hyperparameters": trainRunModelConfig.get(
                            "hyperparameters", {}
                        ),
                    }
            else:
                scoreModelConfig = trainModelConfig.model_config
            if timeFreq == "":
                timeFreq = trainModelConfig.model_time_interval
            if lossFunc == "":
                lossFunc = trainModelConfig.model_loss_function
            ScoreModelSummary(
                score_run_id=scoreRunId,
                model_name=trainModelConfig.model_name,
                model_config=scoreModelConfig,  # TODO: choose best one
                score_loss_function=lossFunc,
                score_time_interval=timeFreq,
                prediction_steps=predSteps,
                prediction_count=predCnt,
                score_trial_status=status,
            )
    else:
        modelParams = scoreParams.get("models", {})
        trainLossFunc = params.get(
            "loss_function", PREDICTION_CONTEXT.default_loss_function
        )
        timeFreq = params.get("time_interval", PREDICTION_CONTEXT.default_time_interval)
        modelConfigsDict = PREDICTION_CONTEXT.default_models_config
        for model in models:
            modelParam = modelParams.get(model, {})
            logger.debug(f"Populating ScoreModelSummary for {model}")
            ScoreModelSummary(
                score_run_id=scoreRunId,
                model_name=model,
                model_config=modelParam.get(
                    "model_config", modelConfigsDict.get(model)
                ),
                score_loss_function=modelParam.get(
                    "score_loss_function", trainLossFunc
                ),
                score_time_interval=params.get("time_interval", timeFreq),
                prediction_steps=predSteps,
                prediction_count=predCnt,
                score_trial_status=status,
            )

    commit()
    return scoreConfig


@db_session
def create_train_run_config(trainParams: dict, username: str) -> TrainRunConfig:
    status = "Created"
    master_params = trainParams.get("master", {})
    resources = master_params.get("resources", {})
    maxJobs = resources.get("max_parallel_pods", 1)
    params = trainParams["training"]
    modelLoc = params.get("model_location", PREDICTION_CONTEXT.S3_PRED_MODEL_PATH)
    trainTask = params.get("train_task_type", PREDICTION_CONTEXT.default_training_task)
    trainTask = trainTask.upper()
    trainName = params["train_name"]
    dataVersion = params.get("data_version", "")
    modelVersion = params.get("model_version", "")
    prediction_steps = params.get("prediction_steps", "")
    if modelVersion is None or modelVersion == "":
        modelComponents = [trainName]
        if dataVersion is not None and dataVersion != "":
            modelComponents.append(dataVersion)
        softVersion = PREDICTION_CONTEXT.kgsa_core_prediction_version
        if softVersion is not None and softVersion != "":
            modelComponents.append(softVersion)
        modelTm = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        modelComponents.append(modelTm)
        modelVersion = "_".join(modelComponents)
    modelType = "ALL"
    models = params.get("model_names", [])
    if models:
        models = [m.upper() for m in models]
    if "ENSEMBLE" in models:
        models = PREDICTION_CONTEXT.ensemble_base_models
        modelType = "ENSEMBLE"

    runConfig = TrainRunConfig(
        train_run_uuid=uuid.uuid4(),
        train_exec_environment=master_params.get(
            "exec_environment", PREDICTION_CONTEXT.default_execution_environment
        ),
        train_data_storage_type=master_params.get(
            "data_storage_type", PREDICTION_CONTEXT.default_data_storage_type
        ),
        train_data_file_type=master_params.get(
            "data_file_type", PREDICTION_CONTEXT.default_file_type
        ),
        train_name=trainName,
        train_description=params.get("train_description", ""),
        train_task_type=trainTask,
        train_job_type=params.get(
            "train_job_type", PREDICTION_CONTEXT.default_train_job_type
        ),
        model_type=modelType,
        target_data_location=params["target_data_location"],
        feature_data_location=params.get("feature_data_location", ""),
        target_feature_mapping_location=params.get(
            "target_feature_mapping_location", ""
        ),
        score_data_location=params.get(
            "score_data_location", PREDICTION_CONTEXT.S3_TRAIN_RESULT_PATH
        ),
        model_version=modelVersion,
        train_data_end_dtm=params["train_data_end_dtm"],
        test_data_end_dtm=params["test_data_end_dtm"],
        validation_data_end_dtm=params["validation_data_end_dtm"],
        data_version=dataVersion,
        model_artifacts_location=modelLoc,
        train_status=status,
        feature_columns=params.get("feature_columns", []),
        jobs_count=maxJobs,
        username=username,
        prediction_steps=prediction_steps,
    )
    commit()

    modelParams = trainParams.get("models", {})
    trainRunId = runConfig.train_run_id
    trainLossFunc = params.get(
        "loss_function", PREDICTION_CONTEXT.default_loss_function
    )
    timeFreq = params.get("time_interval", PREDICTION_CONTEXT.default_time_interval)
    metrics = params.get("metrics", {})

    tuneAlg = ""
    if trainTask == "TUNING":
        tuneAlg = params.get(
            "tuning_algorithm", PREDICTION_CONTEXT.default_tuning_algorithm
        )
        modelConfigsDict = PREDICTION_CONTEXT.default_tune_config
    else:
        modelConfigsDict = PREDICTION_CONTEXT.default_models_config
    for model in models:
        modelParam = modelParams.get(model, {})
        currModelLoc = modelParam.get("model_location", modelLoc)
        if len(currModelLoc) > 0:
            currModelLoc = currModelLoc + f"{model}/"
        TrainRunModelConfig(
            train_run_id=trainRunId,
            model_name=model,
            model_config=modelParam.get(
                "model_config", modelConfigsDict.get(model, None)
            ),  # modelParam["model_config"],  # TODO
            model_loss_function=modelParam.get("model_loss_function", trainLossFunc),
            model_time_interval=modelParam.get("model_time_interval", timeFreq),
            model_artifacts_location=currModelLoc,
            tune_algorithm=modelParam.get("hyperparam_alg", tuneAlg),
            metrics=metrics,
        )
    return runConfig


@db_session(immediate=True, strict=False)
def get_train_run_config(runId: int) -> TrainRunConfig:
    train = TrainRunConfig[runId]
    return train


@db_session(immediate=True)
def get_train_run_config_by_uuid(trainUuid: uuid.UUID, username) -> TrainRunConfig:
    return TrainRunConfig.get(train_run_uuid=trainUuid, username=username)


@db_session(immediate=True)
def get_model_configs_by_train_id(trainRunId: int) -> List:
    trainRunConfig = get_train_run_config(trainRunId)
    return select(m for m in TrainRunModelConfig if m.train_run_id == trainRunConfig)[:]


@db_session
def update_train_status(
    trainRunId: int,
    status: str,
    startTs: datetime = None,
    endTs: datetime = None,
    resultPath: str = None,
) -> TrainRunConfig:
    trainConfig = get_train_run_config(trainRunId)
    trainConfig.train_status = status
    if startTs is not None:
        trainConfig.train_start_ts = startTs
    if endTs is not None:
        trainConfig.train_end_ts = endTs
    if resultPath is not None:
        trainConfig.score_data_location = resultPath
    trainConfig.update_ts = datetime.now()
    commit()
    return trainConfig


@db_session
def create_train_run_trial(
    train_run_model_id: int, trial: TrainModelTrial
) -> TrainRunTrial:
    # TODO trainLossFunc = params.get("loss_function", PREDICTION_CONTEXT.default_loss_function)

    runTrial = TrainRunTrial(
        train_run_model_id=train_run_model_id,
        target_column_name=trial.target_column,
        # TODO feature_columns
        train_model_config=trial.model_config,
        model_loss_function=trial.model_loss_function,
        train_score=trial.train_score,
        test_score=trial.test_score,
        validation_score=trial.validation_score,
        trial_status=trial.trial_status,
        model_trial_location=trial.model_trial_location,
        trial_start_ts=trial.trial_start_ts,
        trial_end_ts=trial.trial_end_ts,
        # TODO Hack to fix nan to nulls. We should implement on pony orm
        metrics=simplejson.loads(simplejson.dumps(trial.metrics, ignore_nan=True)),
    )
    commit()
    return runTrial


@db_session
def get_train_run_trials_by_train_id(trainRunId: int) -> List[TrainRunTrial]:
    trials = TrainRunConfig[trainRunId].models.trials
    logger.debug(f"******trainRunTrials for {trainRunId}: {trials}")
    return trials


@db_session
def get_score_model_summary(runId: int) -> ScoreModelSummary:
    summary = ScoreModelSummary.get(score_model_id=runId)
    return summary


@db_session
def get_score_model_summaries_by_score_id(scoreId: int) -> List:
    score = get_score_run_config(scoreId)
    # logger.info(f"Score: {score}")
    summaries = select(m for m in ScoreModelSummary if m.score_run_id == score)
    return summaries[:]


@db_session
def get_score_run_config(scoreRunId: int) -> ScoreRunConfig:
    scoreConfig = ScoreRunConfig.get(score_run_id=scoreRunId)
    return scoreConfig


@db_session
def get_score_run_config_by_uuid(scoreUuid: uuid.UUID, username) -> ScoreRunConfig:
    score_config = ScoreRunConfig.get(score_run_uuid=scoreUuid, username=username)
    return score_config


@db_session
def get_score_run_configs_by_username(username: str):
    score_configs = ScoreRunConfig.select(lambda u: u.username == username).order_by(
        desc(ScoreRunConfig.score_run_id)
    )
    result = [score_config.to_dict() for score_config in score_configs]
    return result


@db_session
def get_train_run_configs_by_username(username: str):
    train_configs = TrainRunConfig.select(lambda u: u.username == username).order_by(
        desc(TrainRunConfig.train_run_id)
    )
    result = [train_config.to_dict() for train_config in train_configs]
    return result


@db_session
def update_score_status(
    scoreRunId: int,
    status: str,
    startTs: datetime = None,
    endTs: datetime = None,
    resultPath: str = None,
) -> ScoreRunConfig:
    scoreConfig = get_score_run_config(scoreRunId)
    scoreConfig.score_status = status
    if startTs:
        scoreConfig.score_start_ts = startTs
    if endTs:
        scoreConfig.score_end_ts = endTs
    if resultPath:
        scoreConfig.score_data_location = resultPath
    commit()
    return scoreConfig


@db_session
def create_score_model_summary(
    scoreModelId: int,
    col: str,
    status: str,
    forecastedQuantity: List,
    startTs: datetime,
    endTs: datetime,
) -> ScoreModelSummary:
    summary = get_score_model_summary(runId=scoreModelId)
    currModel = ScoreModelSummary(
        score_run_id=summary.score_run_id,
        model_name=summary.model_name,
        model_config=summary.model_config,
        score_loss_function=summary.score_loss_function,
        score_time_interval=summary.score_time_interval,
        prediction_steps=summary.prediction_steps,
        prediction_count=summary.prediction_count,
        score_trial_status=status,
        target_column_name=col,
        forecasted_quantity=forecastedQuantity,
        score_start_ts=startTs,
        score_end_ts=endTs,
    )
    commit()
    return currModel


@db_session
def update_train_model_config_job_name(run_id: int, job_name: str):
    """
    Update TrainRunModelConfig model_job_name column.
    Args:
        run_id: int: TrainRunModelConfig.train_run_model_id.
        job_name: str: train job name.
    """
    train_model_config_obj = TrainRunModelConfig.get(train_run_model_id=run_id)
    train_model_config_obj.set(model_job_name=job_name)
    commit()
    return train_model_config_obj.model_job_name
