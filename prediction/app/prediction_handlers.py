from abc import abstractmethod
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import *
from loguru import logger

from prediction.app.prediction_commands import Command, DataCommand
from prediction.src.utils import read_s3_file

##HANDLER_DICT = {DataCommand: DataHandler, TrainingModel: TSAModeler}


class PredictionHandler:
    """
    The Handler interface declares a method for building the chain of handlers.
    It also declares a method for executing a request.
    """

    @abstractmethod
    def handle(self, request: Command):
        "Code to handle the command"


# class DataHandler(PredictionHandler):
#     def handle(
#         self,
#         request: DataCommand,
#         reader: Callable[[str], Dict[str, pd.DataFrame]] = read_s3_file,
#     ) -> Dict[str, pd.DataFrame]:

#         targets_df = reader(request.target_data_location)
#         features_df = reader(request.feature_data_location)
#         df_mapping = reader(request.target_feature_mapping_location, TS=False)
#         if df_mapping.empty:
#             df_mapping["target_column"] = np.nan
#             df_mapping["feature_column"] = np.nan

#         ## TODO: Bug fix - feature mapping needs to be coded
#         # if train_obj.feature_data_location not in [None, "None"]:
#         #     feature = ut.read_s3_file(train_obj.target_data_location)
#         ## TODO: data split
#         ## TODO: parameterize output
#         df_dict = {}
#         logger.info(targets_df.info())
#         targets_list = list(targets_df.columns)
#         features_list = list(features_df.columns)
#         mapping_count = df_mapping.shape[0]

#         logger.info(f"Data columns: {targets_list}")
#         logger.info(f"Feature columns: {features_list}")
#         logger.info(f"Checking mapping file column names {list(df_mapping.columns)}")
#         logger.info(f"Target to feature mapping count: {mapping_count}")

#         ##targets_list.remove('Date')
#         logger.debug(f"Target df columns: {targets_list}")
#         logger.debug(f"Feature df columns: {features_list}")

#         for target_name in targets_list:
#             df = pd.DataFrame(targets_df[target_name])
#             # Trim leading nulls
#             first_valid = df.first_valid_index()
#             df = df.loc[first_valid:]

#             df = df + np.random.rand(*df.shape) / 10000.0

#             # filter our mapping to just this column
#             df_mapping_target = df_mapping[df_mapping["target_column"] == target_name]
#             # create list of features needed for this target
#             feature_list = list(df_mapping_target["feature_column"])
#             logger.info(
#                 f"Target {target_name} will use only these external features: {feature_list}"
#             )

#             # pull the external features for this target from the source
#             df_f = pd.DataFrame(features_df[feature_list])

#             ##logger.info(df.columns)
#             ##df.Date = pd.to_datetime(df.Date)
#             ##df.set_index('Date')
#             df_dict[target_name] = {"data": df.copy(), "features": df_f.copy()}
#         """
#         df_dict = generate_model_inputs(
#             target, feature, modelName, request.feature_mapping
#         )
#         """
#         ##queue.enque(model_targets_dict)
#         return df_dict

#     def split_train_test_valid(
#         self, df: pd.DataFrame, request: DataCommand
#     ) -> Tuple[pd.DataFrame]:
#         timeUnit = request.model_frequency
#         deltaDict = {
#             "M": relativedelta(months=+1),
#             "D": relativedelta(days=+1),
#             "H": relativedelta(hours=+1),
#         }
#         freq = deltaDict[timeUnit]

#         target_train = df[: request.train_data_end_dtm]
#         target_test = df[: request.test_data_end_dtm + freq]
#         ##df[request.train_data_end_dtm + freq : request.test_data_end_dtm]
#         target_valid = df[: request.validation_data_end_dtm + freq]
#         ##df[request.test_data_end_dtm + freq : request.validation_data_end_dtm]
#         return (target_train, target_test, target_valid)
