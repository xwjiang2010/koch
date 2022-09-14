from typing import Callable, Dict

from prediction.app.prediction_commands import DataCommand
from prediction.app.prediction_handlers import PredictionHandler
from prediction.src.utils import read_s3_file


class LSTMDataPreProcessor(PredictionHandler):
    """ Pre Processing of data for LSTM"""

    def handle(
        self, request: DataCommand, reader: Callable[[str], Dict] = read_s3_file
    ):
        # TODO:
        pass

    def _load_lstm_data():
        # TODO:
        pass

    def _convert_csv_to_ts_dict():
        # TODO:
        pass

    def _upload_json_to_s3():
        # TODO:
        pass
