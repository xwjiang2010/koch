import unittest
from datetime import datetime, time, timedelta
from unittest.mock import patch

from monthdelta import monthdelta

from prediction.app.prediction_commands import DataCommand, TrainingModel
from prediction.app.worker.data_preprocessor.deepar.deepar import DeepARDataPreProcessor
from prediction.app.worker.data_preprocessor.lstm import LSTMDataPreProcessor
from prediction.app.worker.data_preprocessor.tsa import TSADataPreProcessor


class TestDeepARDataPreProcessor(unittest.TestCase):
    """Test class for DeepAR Data Preprocessor"""

    def setUp(self) -> None:
        path = "tests/resources/data/test_single_target.csv"
        self.request = DataCommand(
            target_data_location=path,
            feature_data_location="",
            train_data_end_dtm="1990-01-31",
            test_data_end_dtm="1990-02-03",
            validation_data_end_dtm="1990-03-02",
            model_name="DEEPAR",
            prediction_steps=1,
        )

    @patch("botocore.client.BaseClient._make_api_call")
    def test_deepar_data_preprocessor(self, mock_res):

        command = TrainingModel(
            train_model_id=1,
            train_name="DEEPAR-test",
            train_description="Testing DEEPAR",
            train_job_type="Sequential",
            model_location="s3://",
            model_name="DEEPAR",
            model_config=None,
            model_metric="MAPE",
            model_params={},
            model_time_frequency="M",
            model_hyperparams={},
            modeler=None,
            data_preprocessor=None,
        )

        pre_processor = DeepARDataPreProcessor()
        deepar_data = pre_processor.handle(self.request, command)
        self.assertIsNotNone(deepar_data)
        self.assertEquals(
            deepar_data.get("train"), "s3://prediction-services/prediction_deepAR/train"
        )
        self.assertEquals(
            deepar_data.get("test"), "s3://prediction-services/prediction_deepAR/test"
        )

    def test_resolve_delta(self):
        """
        Test resolve delta method for DeepAR preprocessor.
        Frequencies: M, W, D, H.
        """
        pre_processor = DeepARDataPreProcessor()
        steps = 10

        model_frequency = "MS"
        delta = pre_processor._resolve_delta(model_frequency, steps)
        self.assertEquals(delta, monthdelta(10))

        model_frequency = "D"
        delta = pre_processor._resolve_delta(model_frequency, steps)
        self.assertEquals(delta, timedelta(days=10))

        model_frequency = "W"
        delta = pre_processor._resolve_delta(model_frequency, steps)
        self.assertEquals(delta, timedelta(weeks=10))

        model_frequency = "H"
        delta = pre_processor._resolve_delta(model_frequency, steps)
        self.assertEquals(delta, timedelta(hours=10))


class TestLSTMDataPreProcessor(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_lstm_data_preprocessor(self):
        # TODO
        pass


class TestTSADataPreProcessor(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_tsa_data_preprocessor(self):
        # TODO
        pass
