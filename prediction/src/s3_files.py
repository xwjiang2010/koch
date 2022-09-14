from datetime import datetime

import pandas as pd
from loguru import logger


def read_s3_file(s3_location):
    """Reads the files from the S3 location"""
    try:
        logger.info(f"Reading data from S3 location {s3_location}")
        target = pd.read_csv(s3_location, index_col="Date")
        target.index = pd.to_datetime(target.index)
        return target
    except Exception as e:
        logger.exception(e)
        return None


def create_s3_path(
    s3Folder: str,
    trainName: str,
    trainTs: datetime = datetime.now(),
    suffix: str = "csv",
) -> str:
    if s3Folder[-1] != "/":
        s3Folder = s3Folder + "/"
    trainDtm = trainTs.strftime("%Y-%m-%d-%H.%M.%S")
    fileName = f"{s3Folder}{trainName}_{trainDtm}.{suffix}"
    return fileName


def save_train_result(df: pd.DataFrame, s3Path: str, fileType: str = "csv"):
    # df = pd.DataFrame.from_dict(results, orient="index")

    if fileType.upper() == "CSV":
        logger.info("Saving CSV to S3")
        df.to_csv(s3Path, index=False)
    elif fileType.upper() == "JSON":
        df.to_json(s3Path, index=False)
