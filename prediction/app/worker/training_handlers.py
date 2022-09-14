import sys
from os import path
from threading import Thread
from typing import Tuple, Union

import pandas as pd
import ray
from loguru import logger

from prediction.app.prediction_commands import (
    DataCommand,
    HyperparameterTuning,
    ScoringCommand,
    TrainingModel,
)
from prediction.app.prediction_handlers import PredictionHandler
from prediction.src.nn._deepar import DeepARPredictor
from prediction.src.static import OUTPUT_PATH, PREDICTION_CONTEXT


class TSAModeler(PredictionHandler):
    """
    Instantiate TSA model algorithm and train
    """

    def __init__(
        self,
        target_references_id: int,
    ):
        self.target_references_id = target_references_id

    def handle(self, request: TrainingModel, target: str):
        # TODO :: Get the dataframes from the request
        modelName = request.model_name
        logger.debug(f"{request}")
        params = request.model_params
        hyperparams = request.model_hyperparams
        target_column = target
        with logger.contextualize(target=target_column, model=modelName):
            ##model = globals[modelName](request.model_config)

            ##logger.info(f"**********TSAModeler: Data for {modelName}:")
            ##logger.info(f"{self.data}")

            logger.info(
                f"*******START**TSAModeler: {modelName} model params: {params} model hyperparams: {hyperparams}"
            )

            target_references = ray.get(self.target_references_id)
            target_dict = target_references[target_column]
            data_refs = target_dict["data"]

            df = ray.get(data_refs["train"])
            df_test = ray.get(data_refs["test"]) if "test" in data_refs else None
            df_valid = ray.get(data_refs["valid"]) if "valid" in data_refs else None
            df_features = (
                ray.get(target_dict["features"]) if "features" in target_dict else None
            )

            if modelName == "ARIMA":
                from prediction.src.tsa.arima_predictor import ARIMAPredictor
                arima_param = {"arima_params": params}
                model = ARIMAPredictor(config=arima_param)
            if modelName == "HOLTWINTERS":
                from prediction.src.tsa.holtwinters_predictor import HoltWintersPredictor
                holt_param = {"holt_params": params, "holt_hyperparams": hyperparams}
                logger.info(f"Parameters passed to the Holwinters: {holt_param}")
                model = HoltWintersPredictor(config=holt_param)
                ## Need to remove non-positive values
                col = df.columns[0]
                df = df[df[col] > 0]
                if df_valid is not None:
                    df_valid = df_valid[df_valid[col] > 0]
                if df_test is not None:
                    df_test = df_test[df_test[col] > 0]

            x = (df.index,)
            y = df
            test_x = test_y = None
            valid_x = valid_y = None
            if df_test is not None:
                test_x = (df_test.index,)
                test_y = df_test
            if df_valid is not None:
                valid_x = (df_valid.index,)
                valid_y = df_valid

            if modelName == "PROPHET":
                from prediction.src.tsa.prophet_predictor import ProphetPredictor
                prophet_config = {"prophet_params": params}
                model = ProphetPredictor(config=prophet_config)
                x = df.reset_index().iloc[:, 0]
                y = df.reset_index().iloc[:, 1]
                if df_test is not None:
                    test_x = df_test.reset_index().iloc[:, 0]
                    test_y = df_test.reset_index().iloc[:, 1]
                if df_valid is not None:
                    valid_x = df_valid.reset_index().iloc[:, 0]
                    valid_y = df_valid.reset_index().iloc[:, 1]

            # TODO :: Take this in as user input
            # if self.test_data is not None:
            #     steps = len(self.test_data.index) - len(self.data.index)
            # else:
            #     steps = min(len(self.data.index), 3)
            steps = 3

            def train():
                error = sys.maxsize
                validation_error = sys.maxsize if df_valid is not None else None
                metrics = {}
                # steps = len(x)  ##TODO use configured step
                if steps == 0:
                    return error, validation_error, metrics
                try:
                    ##TODO self.fitted_model = model.fitted_model
                    # x = (self.test_data.index,)
                    # y = self.test_data
                    error = model.score_iteratively(
                        X=test_x, y=test_y, steps=steps, exog=df_features
                    )  ##,error_function=request.model_metric)
                    # Validation

                    if bool(request.metrics) and request.metrics.get("bias", False):
                        metrics["bias"] = model.calc_bias(
                            X=test_x,
                            y=test_y,
                            sample_weights=request.metrics.get("bias", 1),
                            steps=steps,
                            features=df_features,
                        )

                    if df_valid is not None:
                        model.fit(
                            X=test_x, y=test_y, features=df_features
                        )  # self.test_data)
                        validation_error = model.score(
                            X=valid_x, y=valid_y, steps=steps, features=df_features
                        )  # self.validation_data)
                    else:
                        validation_error = None
                    logger.info(f"*****END***TSAModeler: {modelName} score: {error}")
                except Exception as e:
                    logger.error(
                        f"TSAModeler: Error in training with model parameters {params}"
                    )
                    logger.error(f"TSAModeler data causing error: {df}")
                    logger.exception(e)

                return (error, validation_error, metrics)

            def predict():
                try:

                    predCount = request.prediction_count
                    ##forecast = None
                    forecast = model.predict(
                        X=x,
                        y=y,
                        steps=steps,
                        prediction_count=predCount,
                        features=df_features,
                    )
                    ##logger.info(f"******{modelName} forecast: {forecast}")

                    """ if modelName != "PROPHET":
                        forecast = model.predict(
                            X=x, y=y, steps=steps, prediction_count=predCount, features=df_features,
                        )
                    else:
                        forecastSeries = model.predict(X=x, y=y, steps=steps)
                        forecast = forecastSeries.loc["test_prediction"]
                        logger.info(
                            f"******PROPHET forecast: {forecastSeries.loc['test_prediction']}"
                        )
                    """
                    logger.info(
                        f"*****END***TSAModeler: {modelName} Forecast: {forecast}"
                    )
                    return forecast
                except Exception as e:
                    logger.error(f"Error in scoring {modelName} for data")
                    logger.error(f"{df}")
                    logger.exception(e)

                return []

            handlerDict = {
                TrainingModel: train,
                ScoringCommand: predict,
                HyperparameterTuning: train,
            }

            ##handler = handlerDict[type(request)]
            handler = handlerDict.get(type(request), train)
            return handler()


class DeepLearningModeler(PredictionHandler):
    """
    Instantiate and train deep learning trainer
    """

    def __init__(self, trainingPath: str, testPath: str):
        self.train_path = path.join(trainingPath, "train.json")
        self.test_path = path.join(testPath, "valid.json")

    def handle(self, request: TrainingModel, target: str):
        # TODO :: Get the dataframes from the request
        modelName = request.model_name
        config = request.model_params
        ##model = globals[modelName](request.model_config)
        logger.info(f"*******START**TSAModeler: {modelName} training/scoring: {config}")


class DeepARModeler(PredictionHandler):
    """
    Instantiate and train Deep AR trainer.
    """

    train_path = None
    test_path = None
    valid_path = None
    target_column = None
    features = None
    predict_data = None
    model_job_name = None
    tuned_job_name = None

    def __init__(self, target_references_id: int):
        self.target_references_id = target_references_id
        pass

    def load_data(
        self,
        model_command: Union[TrainingModel, ScoringCommand],
        data_command: DataCommand,
    ):
        self.data_command = data_command
        target_references = ray.get(self.target_references_id)
        target_dict = target_references["DEEPAR"]
        data_refs = target_dict["data"]
        data_dict = ray.get(data_refs)

        if isinstance(model_command, TrainingModel):
            self.train_path = path.join(data_dict.get("train"), "train.json")
            self.test_path = path.join(data_dict.get("test"), "test.json")
            self.valid_path = path.join(data_dict.get("valid"), "valid.json")
            self.model_job_name = model_command.job_name

        elif isinstance(model_command, ScoringCommand):
            self.tuned_job_name = model_command.job_name
            for k, v in data_dict.items():
                self.predict_data = v.get("data")

    def handle(self, request: Union[TrainingModel, ScoringCommand], target=None):
        model_name = request.model_name
        params = request.model_params
        hyperparams = request.model_hyperparams
        sagemaker_params = request.model_config.get("sagemaker_params")

        with logger.contextualize(column=self.target_column, model=model_name):
            deepar_params = {}
            deepar_params.update(sagemaker_params)
            deepar_params["tunable_params"] = params
            deepar_params["hyper_params"] = hyperparams

            logger.info(
                f"*******START**DeepARModeler: {model_name} training/scoring: {params}"
            )
            logger.info(f"Parameters passed to DeepAR: {deepar_params}")

            model = DeepARPredictor(
                config=deepar_params, tuned_job_name=self.tuned_job_name
            )

        def train():
            """
            DeepAR Train.
            """
            steps = self.data_command.prediction_steps
            error = sys.maxsize
            if steps == 0:
                return error
            try:
                deepar_train_data = {"train": self.train_path, "test": self.valid_path}

                deepar_model_path = path.join(OUTPUT_PATH, "deepAR/")
                with logger.contextualize(job_name=self.model_job_name):
                    model_fit = model.fit(
                        X=None,
                        y=None,
                        input_channels=deepar_train_data,
                        job_name=self.model_job_name,
                        output_path=deepar_model_path,
                    )
                error = model.score(X=x, y=y, steps=steps)
                logger.info(f"*****----END***DeepARModeler*****score: {error}")
            except Exception as e:
                logger.error(
                    f"*****----DeepARModeler: Error in training with model parameters {params}"
                )
                logger.exception(e)

            return error

        def predict():
            """
            DeepAR Predict.
            """
            try:
                steps = request.prediction_steps
                forecast = model.predict(
                    X=self.predict_data,  # JSONLines DeepAR data instances.
                    pred_steps=steps,
                )
                logger.info(
                    f"*****END***DeepARModeler: {model_name} Forecast: {forecast}"
                )
                return forecast
            except Exception as e:
                logger.error(f"Error in scoring {model_name} for data")
                logger.error(f"{self.predict_data}")
                logger.exception(e)

            return []

        handler_dict = {
            TrainingModel: train,
            ScoringCommand: predict,
            HyperparameterTuning: train,
        }
        handler = handler_dict.get(type(request), train)
        return handler()
