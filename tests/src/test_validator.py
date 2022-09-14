import boto3
import pytest
from moto import mock_s3

from prediction.src.validator import boto_s3_file_validator as boto_s3_file_validator
from prediction.src.validator import find_bucket_key as find_bucket_key


@pytest.fixture
def molex_s3_path():
    MOLEX_DATASET = "s3://prediction-services/data/molex-model-data_2021-06-22.csv"
    return MOLEX_DATASET


@pytest.fixture
def molex_s3_path_invalid():
    MOLEX_DATASET = "s3://prediction-services/data/molex-model-data_2021-06-22"
    return MOLEX_DATASET


@pytest.fixture
def molex_s3_wrong_filename():
    MOLEX_DATASET = "s3://prediction-services/data/molex.csv"
    return MOLEX_DATASET


@pytest.fixture
def fhr_s3_path():
    FHR_DATASET = "s3://kbs-analytics-dev-c3-public/fhr-public/lv-polypropylene-demand-forecast/input/Training/TransformedData/test/TrainingInputData_Transformed_Test.csv"
    return FHR_DATASET


def test_find_bucket_key_valid(molex_s3_path):
    bucket, s3_key, file_name = find_bucket_key(molex_s3_path)
    assert bucket == "prediction-services"
    assert file_name == "molex-model-data_2021-06-22.csv"


def test_find_bucket_key_invalid(molex_s3_path_invalid):
    bucket, s3_key, file_name = find_bucket_key(molex_s3_path_invalid)
    assert bucket == "prediction-services"
    assert file_name != "molex-model-data_2021-06-22.csv"


@mock_s3
def test_boto_s3_file_validator_valid_path(molex_s3_path):
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket="prediction-services")
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.put_object(
        Bucket="prediction-services",
        Key="data/molex-model-data_2021-06-22.csv",
        Body="Some value",
    )
    cls = None
    v = molex_s3_path
    return_val = boto_s3_file_validator(cls, v)
    assert return_val == molex_s3_path


@mock_s3
def test_boto_s3_file_validator_valid_path(molex_s3_path):
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket="prediction-services")
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.put_object(
        Bucket="prediction-services",
        Key="data/molex-model-data_2021-06-22.csv",
        Body="Some value",
    )
    cls = None
    v = molex_s3_path
    return_val = boto_s3_file_validator(cls, v)
    assert return_val == molex_s3_path


@mock_s3
def test_boto_s3_file_validator_exception_catch(molex_s3_wrong_filename):
    with pytest.raises(ValueError):
        conn = boto3.resource("s3", region_name="us-east-1")
        conn.create_bucket(Bucket="prediction-services")
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.put_object(
            Bucket="prediction-services",
            Key="data/molex-model-data_2021-06-22.csv",
            Body="Some value",
        )
        cls = None
        v = molex_s3_wrong_filename
        return_val = boto_s3_file_validator(cls, v)


if __name__ == "__main__":
    test_boto_s3_file_validator_valid_path()
