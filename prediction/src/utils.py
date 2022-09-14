import json
from io import BytesIO, StringIO

import boto3
import jsonlines
import pandas as pd
from loguru import logger
from pandas.core.frame import DataFrame


def read_s3_file(s3_location, TS=True):
    """Reads the files from the S3 location"""
    if not s3_location:
        return pd.DataFrame()

    logger.info(f"Reading data from S3 location {s3_location}")
    if TS:
        df = pd.read_csv(s3_location, index_col="Date")
        df.index = pd.to_datetime(df.index)
    else:
        df = pd.read_csv(s3_location)
    return df


def save_s3_jsonlines(list_dict, bucket, filename, s3_resource=boto3.resource("s3")):
    """Method to write dict in Jsonlines in S3"""
    fp = BytesIO()
    writer = jsonlines.Writer(fp)
    writer.write_all(list_dict)
    s3_resource.Object(bucket, filename).put(Body=fp.getvalue())
    logger.info(f"Writing {len(list_dict)} records to {filename}")
    return f"s3://{bucket}/{filename}"


def read_jsonlines(file_path):
    data = []
    with jsonlines.open(file_path, mode="r") as file:
        for line in file:
            data.append(line)
        return data
