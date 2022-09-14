import unittest

from prediction.app.prediction_commands import TrainingModel
from prediction.app.worker.data_preprocessor.deepar.deepar import DeepARDataPreProcessor
from prediction.app.worker.data_preprocessor.lstm import LSTMDataPreProcessor
from prediction.app.worker.data_preprocessor.tsa import TSADataPreProcessor
from prediction.app.worker.resolvers import (
    resolve_model_command_modeler,
    resolve_model_data_preprocessor,
)
from prediction.app.worker.training_handlers import DeepARModeler, TSAModeler


class TestDataPreProcessorResolver(unittest.TestCase):

    """
    Test Model Data PreProcessor Resolver
    """

    def test_resolve_model_data_preprocessor_arima(self):
        """Test data pre procesoor object for ARIMA"""

        model_command = TrainingModel(
            train_model_id=1,
            train_name="ARIMA-Test",
            train_description="Testing ARIMA Data PreProcessor",
            train_job_type="Sequential",
            model_location="s3://",
            model_name="ARIMA",
            model_config=None,
            model_metric="MAPE",
            model_time_frequency="M",
            model_params={},
            model_hyperparams={},
            modeler=None,
            data_preprocessor=None,
        )
        data_preprocessor = resolve_model_data_preprocessor(model_command)
        self.assertIsNotNone(data_preprocessor)
        assert (data_preprocessor, TSADataPreProcessor)

    def test_resolve_model_data_preprocessor_holtwinter(self):
        """Test data pre procesoor object for HOLTWINTERS"""

        model_command = TrainingModel(
            train_model_id=1,
            train_name="HOLTWINTERS-Test",
            train_description="Testing HOLTWINTERS Data PreProcessor",
            train_job_type="Sequential",
            model_location="s3://",
            model_name="HOLTWINTERS",
            model_config=None,
            model_metric="MAPE",
            model_time_frequency="M",
            model_params={},
            model_hyperparams={},
            modeler=None,
            data_preprocessor=None,
        )
        data_preprocessor = resolve_model_data_preprocessor(model_command)
        self.assertIsNotNone(data_preprocessor)
        assert (data_preprocessor, TSADataPreProcessor)

    def test_resolve_model_data_preprocessor_prophet(self):
        """Test data pre procesoor object for PROPHET"""

        model_command = TrainingModel(
            train_model_id=1,
            train_name="PROPHET-Test",
            train_description="Testing HOLTWINTER Data PreProcessor",
            train_job_type="Sequential",
            model_location="s3://",
            model_name="PROPHET",
            model_config=None,
            model_metric="MAPE",
            model_time_frequency="M",
            model_params={},
            model_hyperparams={},
            modeler=None,
            data_preprocessor=None,
        )
        data_preprocessor = resolve_model_data_preprocessor(model_command)
        self.assertIsNotNone(data_preprocessor)
        assert (data_preprocessor, TSADataPreProcessor)

    def test_resolve_model_data_preprocessor_deepar(self):
        """Test data pre procesoor object for DEEPAR"""

        model_command = TrainingModel(
            train_model_id=1,
            train_name="DEEPAR-Test",
            train_description="Testing DEEPAR Data PreProcessor",
            train_job_type="Sequential",
            model_location="s3://",
            model_name="DEEPAR",
            model_config=None,
            model_metric="MAPE",
            model_time_frequency="M",
            model_params={},
            model_hyperparams={},
            modeler=None,
            data_preprocessor=None,
        )
        data_preprocessor = resolve_model_data_preprocessor(model_command)
        self.assertIsNotNone(data_preprocessor)
        assert (data_preprocessor, DeepARDataPreProcessor)

    def test_resolve_model_data_preprocessor_lstm(self):

        """Test data pre procesoor object for LSTM"""

        model_command = TrainingModel(
            train_model_id=1,
            train_name="LSTM-Test",
            train_description="Testing LSTM Data PreProcessor",
            train_job_type="Sequential",
            model_location="s3://",
            model_name="LSTM",
            model_config=None,
            model_metric="MAPE",
            model_time_frequency="M",
            model_params={},
            model_hyperparams={},
            modeler=None,
            data_preprocessor=None,
        )
        data_preprocessor = resolve_model_data_preprocessor(model_command)
        self.assertIsNotNone(data_preprocessor)
        assert (data_preprocessor, LSTMDataPreProcessor)

    """
       ==========================
        Negative Tests
       ==========================
    """

    def test_negative_resolve_model_data_preprocessor_deepar(self):

        model_command = TrainingModel(
            train_model_id=1,
            train_name="DEEPAR-Test",
            train_description="Testing DEEPAR Data PreProcessor",
            train_job_type="Sequential",
            model_location="s3://",
            model_name="DEEPAR",
            model_config=None,
            model_metric="MAPE",
            model_time_frequency="M",
            model_params={},
            model_hyperparams={},
            modeler=None,
            data_preprocessor=None,
        )
        data_preprocessor = resolve_model_data_preprocessor(model_command)
        self.assertIsNotNone(data_preprocessor)
        self.assertNotEqual(data_preprocessor, TSADataPreProcessor)

    def test_negative_resolve_model_data_preprocessor_lstm(self):

        model_command = TrainingModel(
            train_model_id=1,
            train_name="LSTM-Test",
            train_description="Testing LSTM Data PreProcessor",
            train_job_type="Sequential",
            model_location="s3://",
            model_name="LSTM",
            model_config=None,
            model_metric="MAPE",
            model_time_frequency="M",
            model_params={},
            model_hyperparams={},
            modeler=None,
            data_preprocessor=None,
        )
        data_preprocessor = resolve_model_data_preprocessor(model_command)
        self.assertIsNotNone(data_preprocessor)
        self.assertNotEqual(data_preprocessor, DeepARDataPreProcessor)

    def test_negative_resolve_model_data_preprocessor_lstm(self):

        model_command = TrainingModel(
            train_model_id=1,
            train_name="ARIMA-Test",
            train_description="Testing ARIMA Data PreProcessor",
            train_job_type="Sequential",
            model_location="s3://",
            model_name="LSTM",
            model_config=None,
            model_metric="MAPE",
            model_time_frequency="M",
            model_params={},
            model_hyperparams={},
            modeler=None,
            data_preprocessor=None,
        )
        data_preprocessor = resolve_model_data_preprocessor(model_command)
        self.assertIsNotNone(data_preprocessor)
        self.assertNotEqual(data_preprocessor, TSADataPreProcessor)


class TestModelerResolver(unittest.TestCase):
    """
    Test cases for Resolving Modeler Resolver
    """

    def setUp(self):
        self.model_command = TrainingModel(
            train_model_id=1,
            train_name=None,
            train_description="Testing DEEPAR Data PreProcessor",
            train_job_type="Sequential",
            model_location="s3://",
            model_name=None,
            model_config=None,
            model_metric="MAPE",
            model_time_frequency="M",
            model_params={},
            model_hyperparams={},
            modeler=None,
            data_preprocessor=None,
        )

    def test_resolve_modeler_arima(self):

        self.model_command.train_name = "ARIMA-Test"
        self.model_command.model_name = "ARIMA"

        modeler = resolve_model_command_modeler(self.model_command)
        self.assertIsNotNone(modeler)
        self.assertEqual(modeler, TSAModeler)

    def test_resolve_modeler_prophet(self):

        self.model_command.train_name = "PROPHET-Test"
        self.model_command.model_name = "PROPHET"

        modeler = resolve_model_command_modeler(self.model_command)
        self.assertIsNotNone(modeler)
        self.assertEqual(modeler, TSAModeler)
        pass

    def test_resolve_modeler_holtwinter(self):
        self.model_command.train_name = "HOLTWINTERS-Test"
        self.model_command.model_name = "HOLTWINTERS"

        modeler = resolve_model_command_modeler(self.model_command)
        self.assertIsNotNone(modeler)
        self.assertEqual(modeler, TSAModeler)

    def test_resolve_modeler_deepar(self):
        self.model_command.train_name = "DEEPAR-Test"
        self.model_command.model_name = "DEEPAR"

        modeler = resolve_model_command_modeler(self.model_command)
        self.assertIsNotNone(modeler)
        self.assertEqual(modeler, DeepARModeler)

    def test_resolve_modeler_lstm(self):
        # TODO
        pass
