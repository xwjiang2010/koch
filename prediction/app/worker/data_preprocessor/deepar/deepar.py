import os
from datetime import timedelta
from typing import Callable, Dict, Union

import pandas as pd
from loguru import logger
from monthdelta import monthdelta as m

from prediction.app.prediction_commands import (
    DataCommand,
    ScoringCommand,
    TrainingModel,
)
from prediction.app.prediction_handlers import PredictionHandler
from prediction.app.worker.data_preprocessor.deepar._tsdict_processor import (
    TsDictProcessor,
)
from prediction.src.static import OUTPUT_PATH
from prediction.src.utils import read_s3_file, save_s3_jsonlines


class DeepARDataPreProcessor(PredictionHandler):

    """ Pre-Processing of data for DeepAR"""

    train_start_dtm = None
    train_end_dtm = None
    test_start_dtm = None
    test_end_dtm = None
    validate_start_dtm = None
    validate_end_dtm = None
    predict_end_dtm = None
    model_name = "DEEPAR"
    ts_dict = None
    date_col = "Date"
    model_frequency = "MS"
    s3_bucket = OUTPUT_PATH.split("/")[2].strip()
    prefix = "prediction_deepAR"

    def handle(
        self,
        request: DataCommand,
        command: Union[ScoringCommand, TrainingModel],
        reader: Callable[[str], Dict] = read_s3_file,
    ):
        """
        Main handle or facade method.
        """
        raw_data = self._load_deepar_data(request, reader)
        preprocess = TsDictProcessor(
            raw_data, self.validate_end_dtm, request.model_frequency
        )
        self.ts_dict = preprocess.ts_dict

        if isinstance(command, ScoringCommand):
            data_channel = self._process_deepar_data_predict(request)

        elif isinstance(command, TrainingModel):
            data_channel = self._process_deepar_data_tune(request)

        return data_channel

    def _load_deepar_data(
        self, request: DataCommand, reader: Callable[[str], Dict] = read_s3_file
    ):
        """
        Read CSV and convert into DF dict.
        """
        raw_data = reader(request.target_data_location)

        self._extract_train_test_validate_dtm(request, raw_data)
        data = raw_data.loc[self.train_start_dtm : self.validate_end_dtm].fillna(0)
        logger.info(f'{5 * "="} Raw Data {5 * "="} Shape {data.shape}')
        logger.info(data.columns.tolist())
        return data

    def _extract_train_test_validate_dtm(self, request: DataCommand, targets_df):
        """Extarct start and end timestamp for
        Train, Validate and Test data from DataFrame
           [TRAIN][TEST][VALIDATE]
        """
        self.train_start_dtm = targets_df[: request.train_data_end_dtm].index[0]
        self.train_end_dtm = targets_df[: request.train_data_end_dtm].index[-1]

        self.test_start_dtm = targets_df[
            request.train_data_end_dtm : request.test_data_end_dtm
        ].index[0]
        self.test_end_dtm = targets_df[
            request.train_data_end_dtm : request.test_data_end_dtm
        ].index[-1]

        self.validate_start_dtm = targets_df[request.train_data_end_dtm :].index[0]
        self.validate_end_dtm = targets_df[request.train_data_end_dtm :].index[-1]
        self.predict_end_dtm = targets_df[:].index[-1]

    def _process_deepar_data_tune(self, request: DataCommand):
        """
        Process DeepAR data.
        - Split train, test and validate.
        - Make deepar jsonline instance.
        - Upload deepar jsonlines data to s3.
        """
        self.num_test_windows = self._get_window(self.test_start_dtm, self.test_end_dtm)
        self.num_valid_windows = self._get_window(
            self.validate_start_dtm, self.test_start_dtm
        )
        self.model_frequency = request.model_frequency
        pred_steps = request.prediction_steps

        dataset_config_dict = {
            "train": {
                "end_date": self.validate_start_dtm
                - self._resolve_delta(self.model_frequency, pred_steps),
                "num_test_windows": 0,
                "steps": 0,
            },
            "test": {
                "end_date": self.test_end_dtm,
                "num_test_windows": self.num_test_windows,
                "steps": 0,
            },
            "valid": {
                "end_date": self.test_start_dtm
                - self._resolve_delta(self.model_frequency, pred_steps),
                "num_test_windows": self.num_valid_windows,
                "steps": 0,
            },
            "predict": {
                "end_date": self.train_end_dtm,
                "num_test_windows": 0,
                "steps": request.prediction_steps,
            },
        }
        data_channels = {}
        for name, params in dataset_config_dict.items():
            logger.debug(f"Using: {name=}, {params=}")
            dataset = {
                "path": f"{self.prefix}/{name}/{name}.json",
                "data": self._make_deepar_instances(
                    params["end_date"], params["num_test_windows"], params["steps"]
                ),
            }
            _path = save_s3_jsonlines(dataset["data"], self.s3_bucket, dataset["path"])
            data_channels[name] = os.path.dirname(_path)
            logger.info(f"*******Successfully uploaded {name=} data to: {_path=}")

        """ Creating data channel dict for DeepAR.
            NOTE:
            Here validation data is passed as test data
            and test data is used for test_prediction.
        """
        return data_channels

    def _process_deepar_data_predict(self, request: DataCommand):
        dataset_config_dict = {
            "predict": {
                "end_date": self.predict_end_dtm,
                "num_test_windows": 0,
                "steps": 1,
            },
        }
        data_channels = {}
        for name, params in dataset_config_dict.items():
            logger.debug(f"Using: {name=}, {params=}")
            data = self._make_deepar_instances(
                params["end_date"], params["num_test_windows"], params["steps"]
            )

            dataset = {
                "path": f"{self.prefix}/{name}/{name}.json",
                "data": data,
                "column": data[0].get("index_cols")[0],
            }
            data_channels[dataset["column"]] = {"data": data, "features": None}
        return data_channels

    def _make_deepar_instances(self, end_date, num_test_windows, steps):
        """make train and valid json data from time series dictionary based on a cut off
        datetime at end and number of test windows required.
        """
        json_lines = []
        for group, ts in self.ts_dict.items():
            for t in range(num_test_windows + 1):
                ts_df = ts.loc[
                    : (end_date - self._resolve_delta(self.model_frequency, t))
                ]  # type(ts_df) is pd.DataFrame
                if len(ts_df) > 0:
                    temp = {}
                    temp["index_cols"] = [group]
                    temp["timestamps"] = str(ts_df.index.tolist())
                    temp["start"] = str(ts_df.index.min())
                    temp["target"] = ts_df.tolist()
                    json_lines.append(temp)
        return json_lines

    def _get_window(self, begin_date, end_date):

        """ Return number of evaluation points from end_date """

        delta = 12 * (end_date.year - begin_date.year) + (
            end_date.month - begin_date.month
        )
        if end_date < begin_date:
            raise Exception("Begin date should be less than end date")
        return delta

    def _resolve_delta(self, model_frequency, steps):
        """
        Resolves Delta as per data frequency(time_interval).
        Refer to DeepAR docs for supported DeepAR frequency.
        """
        if model_frequency == "M" or model_frequency == "MS":
            delta = m(steps)
            self.model_frequency = "MS"

        elif model_frequency == "D":
            delta = timedelta(days=steps)
            self.model_frequency = "D"

        elif model_frequency == "W":
            delta = timedelta(weeks=steps)
            self.model_frequency = "W"

        elif model_frequency == "H":
            delta = timedelta(hours=steps)
            self.model_frequency = "H"

        return delta
