import re
import uuid
from datetime import datetime
from os import environ
from typing import Dict, List, Literal, Optional, Union
from uuid import UUID

import boto3
from botocore.errorfactory import ClientError
from pydantic import (
    BaseModel,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    conint,
    conlist,
    root_validator,
    validator,
)

from prediction.src.static import PredictionConfigContext

S3_PRED_PATH = "s3://prediction-services/"
# TODO extend this when we have more supported data locations
FILE_REGEX = re.compile(r"^(s3://|test)([^/]+)/(.*?([^/]+)/?)$")


def validate(data_model, raw_data):
    try:
        return data_model.parse_raw(raw_data)
    except ValidationError as e:
        raise e.json()
    except Exception as e:
        raise e


def s3_validator(cls, v):
    if isinstance(v, str) and FILE_REGEX.fullmatch(v) is None:
        raise ValueError("Invalid file path")
    return v


# SMELL :: This should not exist
def check_if_local(path):
    components = path.split("/")
    if components[0] in ["test", "tests"]:
        return True
    return False


# NOTE :: Maybe use regex
def find_bucket_key(s3_path):
    """
    This is a helper function that given an s3 path such that the path is of
    the form: bucket/key
    It will return the bucket and the key represented by the s3 path
    """
    s3_components = s3_path.split("/")
    bucket = s3_components[2]
    s3_key = ""
    if len(s3_components) > 1:
        s3_key = "/".join(s3_components[3:])
        file_name = s3_components[-1]
    return bucket, s3_key, file_name


# TODO :: Get rid of the test check once test have been updated
def boto_s3_file_validator(cls, v):
    """
    Grab the bucket name and check if the meta data of the file exists.
    Catches 404 and 403 errors froms s3.
    """
    try:
        if check_if_local(v):
            return v
        bucket_name, s3_key, file_name = find_bucket_key(v)
        s3 = boto3.client("s3")
        s3.head_object(Bucket=bucket_name, Key=s3_key)
        return v
    except ClientError as error:
        raise ValueError("Invalid path or lack permissions: {}".format(error))
    except Exception as e:
        raise e


# TODO :: Find a way that doesn't defeat static type checking
ExecType = Literal["Sequential", "Parallel", "Distributed"]
TaskType = Literal["Model", "Tuning"]
DataStorageType = Literal["s3"]
DataStorageFile = Literal["csv", "json"]
StatusType = Literal[
    "Created",
    "Submitted",
    "Running",
    "Completed",
    "Failed",
    "Trained",
    "Tuned",
    "Scored",
    "Cancelled",
]
ModelType = Literal[
    "ARIMA", "HOLTWINTERS", "PROPHET", "ENSEMBLE", "DEEPAR", "LSTM", "MOVINGAVERAGE"
]
TrainTargetType = Literal["Single-Target", "Multi-Target"]
LossFunction = Literal["MAPE"]
TimeInterval = Literal["Y", "M", "D"]
ShortTrendPatterns = Literal["add", "mul", None]
SampleWeights = List[float]


class BaseModelForbiddenExtras(BaseModel):
    class Config:
        extra = "forbid"


class ArimaModelParameters(BaseModelForbiddenExtras):
    p: conlist(conint(ge=0), min_items=1) = [1, 2, 3, 4, 5]
    d: conlist(conint(ge=0), min_items=1) = [0, 1]
    q: conlist(conint(ge=0), min_items=1) = [0, 1, 2]


class HoltwintersParameters(BaseModelForbiddenExtras):
    trend: List[Optional[ShortTrendPatterns]] = ["add", "mul", None]
    damped: List[bool] = [True, False]
    seasonal_periods: Optional[List[PositiveInt]]
    seasonal: List[Optional[ShortTrendPatterns]] = ["add", "mul", None]


class LSTMModelParameters(BaseModelForbiddenExtras):
    FREQ: Literal["MS"] = "MS"
    train_instance_count: List[int] = [1]
    train_instance_type: Literal["ml.m4.10xlarge"] = "ml.m4.10xlarge"
    job_name_prefix: Optional[str]
    early_stopping_type: List[Literal["Auto"]] = ["Auto"]
    max_jobs: int = 30
    max_parallel_jobs: int = 1
    tunable_params: dict = {
        "epochs": [50, 300],
        "context-length": [4, 6],
        "lr": [0.001, 0.01],
        "hidden-dim": [10, 100],
        "num-layers": [1, 2],
        "bias": [True, False],
        "fully-connected": [True, False],
    }


class ProphetModelParameters(BaseModelForbiddenExtras):
    growth: List[Literal["linear", "logistic"]] = ["linear"]
    changepoints: Optional[List[Optional[datetime]]]
    n_changepoints: Optional[List[int]] = [20, 22]
    changepoint_range: List[PositiveFloat] = [0.8, 0.9]
    changepoint_prior_scale: List[PositiveFloat] = [0.05, 0.1]
    yearly_seasonality: Union[List[Literal["auto", True, False]], List[int]] = ["auto"]
    weekly_seasonality: Union[List[Literal["auto", True, False]], List[int]] = ["auto"]
    daily_seasonality: Union[List[Literal["auto", True, False]], List[int]] = ["auto"]
    holidays: Optional[Dict]
    seasonality_mode: List[Literal["additive", "multiplicative"]] = [
        "additive",
        "multiplicative",
    ]
    seasonality_prior_scale: List[float] = [10.0]
    holidays_prior_scale: List[float] = [10.0]
    interval_width: List[float] = [0.8]
    uncertainty_samples: List[int] = [1000]

    @validator("changepoints")
    def validate_mutual_exclusivity_changepoints(cls, changepoints, values):
        if "n_changepoints" in values and values["changepoints"] != [None]:
            raise ValueError("changepoints cannot be set if n_changepoints is set")
        return changepoints

    @validator("n_changepoints")
    def validate_mutual_exclusivity_n_changepoints(cls, n_changepoints, values):
        if (
            "changepoints" in values
            and values["changepoints"] is not None
            and values["changepoints"] != [None]
        ):
            raise ValueError("n_changepoints cannot be set if changepoints is set")
        elif n_changepoints is None:
            return [20]
        return n_changepoints


class DeepARModelParameters(BaseModelForbiddenExtras):
    mini_batch_size: List[int] = [10000]
    epochs: List[int] = [30, 300]
    context_length: List[int] = [3, 10]
    num_cells: List[int] = [1, 20]
    num_layers: List[int] = [1, 5]
    dropout_rate: List[float] = [0.0, 0.10]
    embedding_dimension: List[int] = [1, 15]
    learning_rate: List[float] = [1.0e-3, 1.0e-2]


class DeepARHyperParameters(BaseModelForbiddenExtras):
    time_freq: Literal["M"] = "M"
    early_stopping_patience: int = 5
    cardinality: Literal["auto"] = "auto"
    likelihood: Literal["gaussian"] = "gaussian"
    num_eval_samples: int = 1000
    test_quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class DeepARSagemakerParameters(BaseModelForbiddenExtras):
    num_samples: int = 1000
    pred_steps: int = 3
    FREQ: Literal["MS"] = "MS"
    train_instance_count: int = 1
    train_instance_type: Literal["ml.m4.xlarge"] = "ml.m4.xlarge"
    objective_metric_name: Literal[
        "test:mean_wQuantileLoss"
    ] = "test:mean_wQuantileLoss"
    pred_instance_count: int = 1
    pred_instance_type: Literal["ml.m4.xlarge"] = "ml.m4.xlarge"
    max_jobs: int = 10
    max_parallel_jobs: int = 1
    iam_role: str = "arn:aws:iam::254486207130:role/kbs-analytics-dev-fhr-demand-forecasting-SageMaker-ExecutionRoll"
    tags: List = [
        {
            "Key": "Name",
            "Value": "kgsalasercats@kochind.onmicrosoft.com",
        },
        {"Key": "costcenter", "Value": "54751"},
        {"Key": "blc", "Value": "1559"},
    ]


class ModelDisplay(BaseModelForbiddenExtras):
    disp: Literal[0, 1] = 0


class HoltwintersHyperparameters(BaseModelForbiddenExtras):
    optimized: bool = True
    use_boxcox: bool = True
    remove_bias: bool = True


class LSTMHyperparameters(BaseModelForbiddenExtras):
    batch_size: int = 10000
    test_batch_size: int = 10000
    log_interval: int = 1000
    dropout_eval: List[PositiveFloat] = [0.2]
    normalized: bool = True
    test_loss_func: Literal["mape"] = "mape"
    loss_func: Literal["mape"] = "mape"


class MovingAverageHyperparameters(BaseModelForbiddenExtras):
    number_of_periods: int = 0
    imputation: bool = True


class ArimaModelConfig(BaseModelForbiddenExtras):
    parameters: ArimaModelParameters = ArimaModelParameters()
    hyperparameters: ModelDisplay = ModelDisplay()


class HoltWintersModelConfig(BaseModelForbiddenExtras):
    parameters: HoltwintersParameters = HoltwintersParameters()
    hyperparameters: HoltwintersHyperparameters = HoltwintersHyperparameters()


class ProphetModelConfig(BaseModelForbiddenExtras):
    parameters: ProphetModelParameters = ProphetModelParameters()
    hyperparameters: Optional[dict] = {}


class DeepARModelConfig(BaseModelForbiddenExtras):
    parameters: DeepARModelParameters = DeepARModelParameters()
    hyperparameters: DeepARHyperParameters = DeepARHyperParameters()
    sagemakerparameters: DeepARSagemakerParameters = DeepARSagemakerParameters()


class LSTMModelConfig(BaseModelForbiddenExtras):
    parameters: LSTMModelParameters = LSTMModelParameters()
    hyperparameters: LSTMHyperparameters = LSTMHyperparameters()


class MovingAverageModelConfig(BaseModelForbiddenExtras):
    hyperparameters: MovingAverageHyperparameters = MovingAverageHyperparameters()


class ArimaModel(BaseModelForbiddenExtras):
    model_name: Literal["ARIMA"] = "ARIMA"
    model_time_interval: TimeInterval = "M"
    model_config: ArimaModelConfig = ArimaModelConfig()


class HoltWintersModel(BaseModelForbiddenExtras):
    model_name: Literal["HOLTWINTERS"] = "HOLTWINTERS"
    model_time_interval: TimeInterval = "M"
    model_config: HoltWintersModelConfig = HoltWintersModelConfig()


class ProphetModel(BaseModelForbiddenExtras):
    model_name: Literal["PROPHET"] = "PROPHET"
    model_time_interval: TimeInterval = "M"
    model_config: ProphetModelConfig = ProphetModelConfig()


class DeepARModel(BaseModelForbiddenExtras):
    model_name: Literal["DEEPAR"] = "   DEEPAR"
    model_time_interval: TimeInterval = "M"
    model_config: DeepARModelConfig = DeepARModelConfig()


class LSTMModel(BaseModelForbiddenExtras):
    model_name: Literal["LSTM"] = "LSTM"
    model_time_interval: TimeInterval = "M"
    model_config: LSTMModelConfig = LSTMModelConfig()


class MovingAverageModel(BaseModelForbiddenExtras):
    model_name: Literal["MOVINGAVERAGE"] = "MOVINGAVERAGE"
    model_time_interval: TimeInterval = "M"
    model_config: MovingAverageModelConfig = MovingAverageModelConfig()


class Master(BaseModel):
    exec_environment: Optional[ExecType]
    data_storage_type: Optional[DataStorageType]
    data_file_type: Optional[DataStorageFile]
    resources: Optional[Dict]


class Models(BaseModelForbiddenExtras):
    ARIMA: Optional[ArimaModel] = None
    HOLTWINTERS: Optional[HoltWintersModel] = None
    PROPHET: Optional[ProphetModel] = None
    DEEPAR: Optional[DeepARModel] = None
    LSTM: Optional[LSTMModel] = None
    MOVINGAVERAGE: Optional[MovingAverageModel] = None


class BiasConfig(BaseModelForbiddenExtras):
    name: str = "bias"
    config: Optional[SampleWeights]


####################### Training Validator #######################
class TrainType(BaseModelForbiddenExtras):

    model_names: List[ModelType]
    target_data_location: str
    test_data_end_dtm: datetime
    train_data_end_dtm: datetime
    train_name: str
    validation_data_end_dtm: datetime

    data_version: Optional[str]
    feature_data_location: str = ""
    metrics: Optional[List[BiasConfig]]
    model_artifacts_location: str = ""
    model_version: str = environ.get("IMAGE_TAG", "development-model")
    score_data_location: str = f"{S3_PRED_PATH}score/"

    target_data_location: str
    target_feature_mapping_location: Optional[str]

    train_data_file_type: DataStorageFile = "csv"
    train_data_storage_type: DataStorageType = "s3"
    train_description: str = ""
    train_exec_environment: ExecType = "Sequential"
    train_job_type: TrainTargetType = "Multi-Target"
    train_run_uuid: UUID = uuid.uuid4()
    train_task_type: TaskType = "Model"
    hyperparam_alg: Optional[str]
    loss_function: LossFunction = "MAPE"
    model_location: Optional[str]
    time_interval: Optional[TimeInterval]
    prediction_steps: Optional[int] = PredictionConfigContext.default_prediction_steps

    @validator("prediction_steps")
    def validate_pred_steps(cls, v):
        return v or PredictionConfigContext.default_prediction_steps

    @validator(
        "test_data_end_dtm", "train_data_end_dtm", "validation_data_end_dtm", pre=True
    )
    def parse_dates(cls, v):
        if isinstance(v, str):
            try:
                return datetime.strptime(v, "%Y-%m-%d")  # maybe needs try/except
            except:
                raise ValidationError("Unable to parse date string")
        return v

    @validator("target_data_location", "model_location")
    def validate_data_location(cls, v):
        return s3_validator(cls, v)

    @validator("target_data_location", "model_location")
    def validate_data_path_access(cls, v):
        return boto_s3_file_validator(cls, v)


class TrainInput(BaseModel):
    models: Optional[Models]
    master: Optional[Master]
    training: TrainType

    class Config:
        schema_extra = {
            "training": {
                "train_name": "model-minimum-config-jupyter",
                "target_data_location": "s3://prediction-services/test_single_target.csv",
                "train_data_end_dtm": "1990-01-31",
                "test_data_end_dtm": "1990-02-03",
                "validation_data_end_dtm": "1990-03-02",
                "model_names": ["ARIMA", "HOLTWINTERS"],
            }
        }
        extra = "forbid"


####################### Scoring Validator #######################
class ScoringType(BaseModelForbiddenExtras):
    score_name: str
    score_data_location: str = f"{S3_PRED_PATH}score/"
    score_loss_function: LossFunction = "MAPE"
    model_names: List[ModelType]
    target_data_location: str
    prediction_steps: int
    prediction_count: int
    feature_data_location: Optional[str]
    target_feature_mapping_location: Optional[str]
    train_run_id: Optional[UUID]

    @validator("target_data_location")
    def validate_data_location(cls, v):
        return s3_validator(cls, v)


class PredictionInput(BaseModelForbiddenExtras):
    models: Optional[Models]
    master: Optional[Master]
    training: Optional[TrainType]
    scoring: ScoringType


# Response Types
class ServiceStatus(BaseModelForbiddenExtras):
    status: Literal["OK", "Unhealthy"]


class StatusResponse(BaseModelForbiddenExtras):
    runId: UUID
    status: StatusType
    status_message: Optional[str] = ""

    @root_validator(pre=True)
    def extract_status(cls, values):
        if " " in (status := values.get("status")):
            values["status_message"] = status
            values["status"] = status.split(" ")[0]
        return values


class TrainStatusResponse(StatusResponse):
    trainStartTs: Optional[datetime]
    updateTs: datetime

    class Config:
        extra = "forbid"


class TrainResultResponse(TrainStatusResponse):
    resultLocation: str
    trainEndTs: Optional[datetime]

    class Config:
        extra = "forbid"


class PredictResultResponse(TrainResultResponse):
    scoreStartTs: Optional[datetime]
    scoreEndTs: Optional[datetime]

    class Config:
        extra = "forbid"


class PredictDelete(BaseModelForbiddenExtras):
    runId: UUID
    status: Literal["Killed the job successfully"]

    class Config:
        extra = "forbid"


class TrainingDelete(BaseModelForbiddenExtras):
    runId: UUID
    status: Literal["Killed the job successfully", "Deleted the endpoint successfully"]

    class Config:
        extra = "forbid"


class PredictResultList(BaseModel):
    __root__: List[PredictResultResponse]


class TrainResultList(BaseModel):
    __root__: List[TrainResultResponse]
