"""
This module contains methods to dispatch modelers
and data Pre-Processor based on different model type
Resolver are based on dispatcher pattern.
"""
from typing import Union

from prediction.app.prediction_commands import HyperparameterTuning, TrainingModel
from prediction.app.worker.data_preprocessor.deepar.deepar import DeepARDataPreProcessor
from prediction.app.worker.data_preprocessor.lstm import LSTMDataPreProcessor
from prediction.app.worker.data_preprocessor.tsa import TSADataPreProcessor
from prediction.app.worker.training_handlers import (
    DeepARModeler,
    DeepLearningModeler,
    TSAModeler,
)


def resolve_model_command_modeler(
    model_command: Union[TrainingModel, HyperparameterTuning]
):
    """
    Resolves 'modeler' based on
    model_command--> model_name
    (TrainingModel/HyperparameterTuning)
    and set model_command--> modeler field.
    """
    if model_command.model_name in ["ARIMA", "PROPHET", "HOLTWINTERS"]:
        model_command.modeler = TSAModeler
    elif model_command.model_name == "DEEPAR":
        model_command.modeler = DeepARModeler
    elif model_command.model_name == "LSTM":
        model_command.modeler = DeepLearningModeler

    return model_command.modeler


def resolve_model_data_preprocessor(
    model_command: Union[TrainingModel, HyperparameterTuning]
):
    """
    Resolves 'data_preprocessor' based on
    model_command--> model_name
    (TrainingModel/HyperparameterTuning)
    and set model_command--> data_preprocessor field.
    """
    if model_command.model_name in ["ARIMA", "PROPHET", "HOLTWINTERS"]:
        model_command.data_preprocessor = TSADataPreProcessor
    elif model_command.model_name in ["DEEPAR"]:
        model_command.data_preprocessor = DeepARDataPreProcessor
    elif model_command.model_name in ["LSTM"]:
        model_command.data_preprocessor = LSTMDataPreProcessor

    return model_command.data_preprocessor
