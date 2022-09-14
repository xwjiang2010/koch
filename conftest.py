import os
import tempfile
from pathlib import Path
from unittest import mock

# Third Party
import pytest
from pony.orm import db_session

from prediction.src.db.prediction_database import db

host = "127.0.0.1"


@pytest.fixture()
def tempdir():
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.dict(
            os.environ, {"OUTPUT_PATH": td, "INPUT_PATH": "./tests/resources/data/"}
        ):
            yield td


@pytest.fixture(autouse=True)
def set_tempdir(monkeypatch, tempdir):
    temp_path = Path(tempdir)
    for directory in ["model", "score", "train"]:
        (temp_path / directory).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "prediction.src.static.PREDICTION_CONTEXT.S3_PRED_PATH", tempdir
    )
    monkeypatch.setattr(
        "prediction.src.static.PREDICTION_CONTEXT.S3_DATA_PATH", tempdir
    )
    monkeypatch.setattr(
        "prediction.src.static.PREDICTION_CONTEXT.S3_PRED_MODEL_PATH",
        f"{tempdir}/model",
    )
    monkeypatch.setattr(
        "prediction.src.static.PREDICTION_CONTEXT.S3_SCORE_DATA_PATH",
        f"{tempdir}/score",
    )
    monkeypatch.setattr(
        "prediction.src.static.PREDICTION_CONTEXT.S3_TRAIN_RESULT_PATH",
        f"{tempdir}/train",
    )
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.dict(
            os.environ, {"OUTPUT_PATH": td, "INPUT_PATH": "./tests/resources/data/"}
        ):
            yield td


@pytest.fixture(autouse=True)
@db_session
def truncate_db():
    tables = db.select(
        "select tablename from pg_tables where schemaname='kgsa-core-prediction-v1'"
    )
    for table in tables:
        db.execute(
            f'TRUNCATE "kgsa-core-prediction-v1".{table} RESTART IDENTITY CASCADE'
        )
