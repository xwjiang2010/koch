from pathlib import Path

import pytest
from pony.orm import db_session

from prediction.src.db.prediction_database import db

test_dir = Path(__file__).parent
sql_scripts_dir = test_dir / "resources/sql/"


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@db_session
def reset_sequences():
    db.execute(
        "SELECT setval('postgres.kgsa-core-prediction-v1.train_run_config_train_run_id_seq', "
        '            (SELECT MAX(train_run_id) FROM "kgsa-core-prediction-v1".train_run_config) + 1);'
        "SELECT setval('postgres.kgsa-core-prediction-v1.train_run_model_config_train_run_model_id_seq',"
        '            (SELECT MAX(train_run_model_id) FROM "kgsa-core-prediction-v1".train_run_model_config) + 1);'
        "SELECT setval('postgres.kgsa-core-prediction-v1.train_run_trial_train_run_trial_id_seq',"
        '            (SELECT MAX(train_run_trial_id) FROM "kgsa-core-prediction-v1".train_run_trial) + 1);'
        "SELECT setval('postgres.kgsa-core-prediction-v1.score_run_config_score_run_id_seq',"
        '            (SELECT MAX(score_run_id) FROM "kgsa-core-prediction-v1".score_run_config) + 1);'
        "SELECT setval('postgres.kgsa-core-prediction-v1.score_model_summary_score_run_id_seq',"
        '            (SELECT MAX(score_run_id) FROM "kgsa-core-prediction-v1".score_model_summary) + 1);'
        "SELECT setval('postgres.kgsa-core-prediction-v1.score_model_summary_score_model_id_seq',"
        '            (SELECT MAX(score_model_id) FROM "kgsa-core-prediction-v1".score_model_summary) + 1);'
    )


@db_session
def seed_table(table):
    file_path = sql_scripts_dir / f"{table}.sql"
    with file_path.open() as sql_script:
        sql = sql_script.read()
        db.execute(sql.replace("NaN", "'NaN'"))


@pytest.fixture(autouse=True)
@db_session
def seed_data():
    # Set up Model Params Table
    for table in [
        "model_parameter",
        "train_run_config",
        "train_run_model_config",
        "train_run_trial",
        "score_run_config",
        "score_model_summary",
    ]:
        seed_table(table)
    reset_sequences()

    # Set up Train Table
