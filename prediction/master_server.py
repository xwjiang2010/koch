import os
from datetime import datetime
from uuid import UUID

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from loguru import logger

import prediction.src.db.prediction_database as pred_db
from prediction.app.prediction_manager import PredictionJobHandler, PredictionMediator
from prediction.log_config import setup_logging
from prediction.src.nn.sagemaker_hpo import SageMakerTrainer
from prediction.src.validator import (
    PredictDelete,
    PredictionInput,
    PredictResultList,
    PredictResultResponse,
    ServiceStatus,
    StatusResponse,
    TrainingDelete,
    TrainInput,
    TrainResultList,
    TrainResultResponse,
)

thread_list = []

app = FastAPI(
    title="KGSA Predictions Service",
    description="Provides model training and prediction services",
    version=os.environ.get("IMAGE_TAG", "development"),
)

thread_list = []


@app.post(
    "/training",
    response_model=StatusResponse,
    response_model_exclude_unset=True,
    summary="Train or Tune a Model",
    tags=["Train"],
)
async def train(body: TrainInput, x_consumer_username: str = Header(None)):
    """## This end point is utilized to train or tune a model. Training can be divided into two categories:

       - **Model**: Training one or more models based on single parameter set (either default or user provided)
       - **Tuning**: Train one or more models in automated mode (population based training)

    ### Training runs will require the following parameters to be populated:

       - **train_name**: Used to identify training job and used in model version
       - **target_data_location**: S3 (only S3 is supported) location of the CSV file containing the training data
       - **train_data_end_dtm**: Last inclusive day to be used for training
       - **test_data_end_dtm**: Last inclusive day to be used for testing
       - **validation_data_end_dtm**: Last day or the remaining data
       - **model_names**: List of models to be trained on the data. Valid values includes: Arima, Prophet, Holtwinters, DeepAR, LSTM, and Ensemble
       - Currently, the following resource can be defined: maximum number of parallel pods to use in the job (i.e. "max_parallel_pods": 3).

    ### Since this is an implementation detail.

        Deprecated: Defaults are being moved over to the api layer.

    TODO: Remove
       - The default training task is "Model" which trains on the data using a single default set of model parameters
       - The code will run in Sequential mode, i.e., on a single core
       - The default loss function is MAPE
       - The default prediction frequency is Monthly
    """
    with logger.contextualize(user=x_consumer_username):
        logger.debug(f"Running training: {body}")

        mediator = PredictionMediator(
            body.dict(exclude_none=True), username=x_consumer_username
        )
        runId = mediator.runConfig.train_run_uuid
        with logger.contextualize(run_id=runId):
            status = mediator.handle_command()
        return {"runId": runId, "status": "Submitted" if status else "Failed"}


# TODO :: Consider combining this and the train status endpoint
@app.get(
    "/trainings/{model_training_run_id}",
    response_model=TrainResultResponse,
    response_model_exclude_unset=True,
    summary="Get Training Result",
    tags=["Train"],
)
async def fetch_train_results(
    model_training_run_id: UUID, x_consumer_username: str = Header(None)
):
    """## This service is utilized to return the results of a training or tuning run.

    If a unique job id is specified within the end point, then the status of that specific job is returned.
    If a unique Id is not specified, all the jobs associated with the user calling the service are returned.

    ### The following attributes are returned:

    - **runid**: The unique identifer of the job.
    - **status**: The status of the train or tune job. Valid values are: Created, Submitted, Running, Completed, Failed, Trained, Tuned, Scored
    - **resultLocation**: The S3 location where the results of the train or tune job are stored.
    - **trainStartTs**: The timestamp of when the train/tune service was called.
    - **trainEndTs**: The timestamp of when the train/tune service completed running.
    """
    with logger.contextualize(user=x_consumer_username):
        logger.debug(f"Fetching training results of {model_training_run_id}")
        trainRunConfig = pred_db.get_train_run_config_by_uuid(
            model_training_run_id, x_consumer_username
        )
        if trainRunConfig is None:
            raise HTTPException(status_code=404, detail="Train run ID not found")
        logger.debug(
            f"status of {model_training_run_id} = {trainRunConfig.train_status}"
        )

        result = {
            "runId": trainRunConfig.train_run_uuid,
            "status": trainRunConfig.train_status,
            "resultLocation": trainRunConfig.score_data_location,
            "trainStartTs": trainRunConfig.train_start_ts,
            "trainEndTs": trainRunConfig.train_end_ts,
            "updateTs": trainRunConfig.update_ts,
        }
        logger.debug(f"Training results of {model_training_run_id}: {result}")
        return result


@app.post(
    "/predictions",
    response_model=StatusResponse,
    response_model_exclude_unset=True,
    summary="Predict",
    tags=["Predict"],
)
async def predict(body: PredictionInput, x_consumer_username: str = Header(None)):

    """## This end point is utilized to make a prediction/inference on a time series.

    ### Prediction runs will require the following parameters to be populated:

       - **score_name**: Used to identify scoring/prediction job and used in model version
       - **target_data_location**: The S3 (only S3 is supported) location of the CSV file containing the prediction input data
       - **score_data_location**: The S3 location of where the resulting CSV file from the prediction job will be stored.
       - **model_names**: List of predictive models that will be utilized to make the prediction. Valid values includes: Arima, Prophet, Holtwinters, DeepAR, LSTM, and Ensemble
       - **prediction_steps**: The number of prediction steps/time series steps to make a prediction for. The prediction starts from the following month.

       **NOTE**:

       - If a train_run_id is provided it will utilize a previously trained model, if one is not provided, it will utilize a new model.
       - Within each model class, users can provide the specific model parameters and hyperparameters to utilize within a prediction job. If none are provided, default values will be used.
       - Currently, the following resource can be defined: maximum number of parallel pods to use in the job (i.e. "max_parallel_pods": 3).

    """
    with logger.contextualize(user=x_consumer_username):
        logger.debug(f"Generating predictions: {body}")
        mediator = PredictionMediator(
            params=body.dict(exclude_unset=True),
            isTrain=False,
            username=x_consumer_username,
        )
        status = mediator.predict()
        if status == "Failed":  # To throw a 400 status code when status is "Failed"
            return_status = {
                "runId": str(mediator.scoreConfig.score_run_uuid),
                "status": "Failed",
            }
            raise HTTPException(status_code=400, detail=return_status)
        else:
            return {
                "runId": mediator.scoreConfig.score_run_uuid,
                "status": "Submitted",  # Returning this when status is not "Failed"
            }


# TODO :: Consider combining this and the prediction status endpoint
@app.get(
    "/predictions/{model_predict_id}",
    response_model=PredictResultResponse,
    response_model_exclude_unset=True,
    summary="Get Prediction Results",
    tags=["Predict"],
)
async def fetch_predictions(
    model_predict_id: UUID, x_consumer_username: str = Header(None)
):
    """## This service is utilized to return the results of a prediction run.

    If a unique job id is specified within the end point, then the status of that specific job is returned.
    If a unique Id is not specified, all the jobs associated with the user calling the service are returned.

    ### The following attributes are returned:

    - **runid**: The unique identifer of the job.
    - **status**: The status of the prediction job. Valid values are: Created, Submitted, Running, Completed, Failed, Trained, Tuned, Scored
    - **status_message**: This describes casual factors for the status (for example for failures it will indicate the failure reason)
    - **resultLocation**: The S3 location where the results of the prediction are stored.
    - **scoreStartTs**: The timestamp of when the prediction service was called.
    - **scoreEndTs**: The timestamp of when the prediction service completed running.
    """
    with logger.contextualize(user=x_consumer_username):
        score = pred_db.get_score_run_config_by_uuid(
            model_predict_id, x_consumer_username
        )
        if score is None:
            raise HTTPException(status_code=404, detail="Prediction ID not found")
        result = {
            "runId": score.score_run_uuid,
            "status": score.score_status,
            "resultLocation": score.score_data_location,
            "scoreStartTs": score.score_start_ts,
            "scoreEndTs": score.score_end_ts,
            "updateTs": score.update_ts,
        }
        return result


@app.get(
    "/predictions",
    response_model=PredictResultList,
    response_model_exclude_unset=True,
    tags=["Predict"],
)
async def predictions(x_consumer_username: str = Header(None)):
    """## This service is utilized to return the results of a prediction run.

    If a unique job id is specified within the end point, then the status of that specific job is returned.
    If a unique Id is not specified, all the jobs associated with the user calling the service are returned.

    ### The following attributes are returned:

    - **runid**: The unique identifer of the job.
    - **status**: The status of the prediction job. Valid values are: Created, Submitted, Running, Completed, Failed, Trained, Tuned, Scored
    - **resultLocation**: The S3 location where the results of the prediction are stored.
    - **scoreStartTs**: The timestamp of when the prediction service was called.
    - **scoreEndTs**: The timestamp of when the prediction service completed running.
    """
    with logger.contextualize(user=x_consumer_username):
        predictions_list = pred_db.get_score_run_configs_by_username(
            x_consumer_username
        )
        return [
            {
                "runId": score["score_run_uuid"],
                "status": score["score_status"],
                "resultLocation": score["score_data_location"],
                "scoreStartTs": score["score_start_ts"],
                "scoreEndTs": score["score_end_ts"],
                "updateTs": score["update_ts"],
            }
            for score in predictions_list
        ]


@app.get(
    "/trainings",
    response_model=TrainResultList,
    response_model_exclude_unset=True,
    tags=["Train"],
)
async def trainings(x_consumer_username: str = Header(None)):
    """## This service is utilized to return the results of a training or tuning run.

    If a unique job id is specified within the end point, then the status of that specific job is returned.
    If a unique Id is not specified, all the jobs associated with the user calling the service are returned.

    ### The following attributes are returned:

    - **runid**: The unique identifer of the job.
    - **status**: The status of the train or tune job. Valid values are: Created, Submitted, Running, Completed, Failed, Trained, Tuned, Scored
    - **resultLocation**: The S3 location where the results of the train or tune job are stored.
    - **trainStartTs**: The timestamp of when the train/tune service was called.
    - **trainEndTs**: The timestamp of when the train/tune service completed running.
    """
    with logger.contextualize(user=x_consumer_username):
        trainings_list = pred_db.get_train_run_configs_by_username(x_consumer_username)
        return [
            {
                "runId": train_run_config["train_run_uuid"],
                "status": train_run_config["train_status"],
                "resultLocation": train_run_config["score_data_location"],
                "trainStartTs": train_run_config["train_start_ts"],
                "trainEndTs": train_run_config["train_end_ts"],
                "updateTs": train_run_config["update_ts"],
            }
            for train_run_config in trainings_list
        ]


@app.delete(
    "/predictions/{model_predict_id}",
    response_model=PredictDelete,
    response_model_exclude_unset=True,
    summary="Kill Prediction Jobs",
    tags=["Predict"],
)
async def delete_predictions(
    model_predict_id: UUID, x_consumer_username: str = Header(None)
):
    """## This service is utilized to kill a prediction job.

    If a valid unique job id is specified within the end point, then if it exists,it is deleted.
    If an invalid unique job id is specified within the end point, then an error is thrown.

    ### The following attributes are returned:

    - **runid**: The unique identifer of the job.
    - **status**: The status of deletion of prediction job.
    """

    with logger.contextualize(user=x_consumer_username):
        score_run_config = pred_db.get_score_run_config_by_uuid(
            model_predict_id, username=x_consumer_username
        )
        if score_run_config is None:
            raise HTTPException(status_code=404, detail="Prediction run ID not found")

        modelConfig = pred_db.get_model_configs_by_train_id(
            score_run_config.train_run_id
        )
        model_names = []

        for model_config in modelConfig:
            model_names.append(model_config.model_name)

        # DeepAR Jobs Deletion

        if "DEEPAR" in model_names:
            flag = model_names.index("DEEPAR")
            sagemaker_job_name = modelConfig[flag].model_job_name

            job_status = SageMakerTrainer.get_job_status(sagemaker_job_name)
            if job_status == "Completed":
                SageMakerTrainer.remove_endpoint(sagemaker_job_name)

        # Other Model Jobs Deletion

        job_handler = PredictionJobHandler(
            model_predict_id, score_run_config.score_run_id, user=x_consumer_username
        )
        status_of_job = job_handler.status_of_job()
        if status_of_job == "Untraceable":
            detail_for_exception = {
                "runId": str(model_predict_id),
                "status": "Could not get the status of the job",
            }
            raise HTTPException(status_code=404, detail=detail_for_exception)
        status_of_deletion = job_handler.delete_active_job(status_of_job, False)
        if status_of_deletion == "Failed":
            detail_for_exception = {
                "runId": str(model_predict_id),
                "status": "Could not kill the job",
            }
            raise HTTPException(status_code=404, detail=detail_for_exception)
        elif status_of_deletion == "Success":
            result = {
                "runId": score_run_config.score_run_uuid,
                "status": "Killed the job successfully",
            }
        return result


@app.delete(
    "/trainings/{model_training_run_id}",
    response_model=TrainingDelete,
    response_model_exclude_unset=True,
    summary="Kill Training/Tuning Jobs",
    tags=["Train"],
)
async def delete_trainings(
    model_training_run_id: UUID, x_consumer_username: str = Header(None)
):
    """## This service is utilized to kill a training/tuning job.

    If a valid unique job id is specified within the end point, then if it exists,it is deleted.
    If an invalid unique job id is specified within the end point, then an error is thrown.

    ### The following attributes are returned:

    - **runid**: The unique identifer of the job.
    - **status**: The status of deletion of training/tuning job.
    """
    with logger.contextualize(user=x_consumer_username):
        trainRunConfig = pred_db.get_train_run_config_by_uuid(
            model_training_run_id, username=x_consumer_username
        )
        if trainRunConfig is None:
            raise HTTPException(status_code=404, detail="Train run ID not found")

        # Sagemaker vs Normal Jobs Differentiation
        modelConfig = pred_db.get_model_configs_by_train_id(trainRunConfig.train_run_id)
        model_names = []

        for model_config in modelConfig:
            model_names.append(model_config.model_name)

        # DeepAR Jobs Deletion

        if "DEEPAR" in model_names:
            flag = model_names.index("DEEPAR")
            sagemaker_job_name = modelConfig[flag].model_job_name

            job_status = SageMakerTrainer.get_job_status(sagemaker_job_name)
            if job_status == "Completed":
                endpoint_deletion_status = SageMakerTrainer.remove_endpoint(
                    sagemaker_job_name
                )
                if endpoint_deletion_status == False:
                    detail_for_exception = {
                        "runId": str(trainRunConfig.train_run_uuid),
                        "status": "Could not delete the endpoint.",
                    }
                    raise HTTPException(status_code=404, detail=detail_for_exception)

                else:
                    result = {
                        "runId": str(trainRunConfig.train_run_uuid),
                        "status": "Deleted the endpoint successfully",
                    }
                    return result

            sagemaker_job_deletion_status = SageMakerTrainer.stop_running_process(
                sagemaker_job_name
            )

            if sagemaker_job_deletion_status == "STOPPED":
                detail_for_exception = {
                    "runId": str(trainRunConfig.train_run_uuid),
                    "status": "The job is currently not running.",
                }
                raise HTTPException(status_code=404, detail=detail_for_exception)

            elif sagemaker_job_deletion_status == "UNTRACEABLE":
                detail_for_exception = {
                    "runId": str(trainRunConfig.train_run_uuid),
                    "status": "Could not get the status of the job",
                }
                raise HTTPException(status_code=404, detail=detail_for_exception)

            result = {
                "runId": str(trainRunConfig.train_run_uuid),
                "status": "Killed the job successfully",
            }

        # Other Model Jobs Deletion

        else:

            job_handler = PredictionJobHandler(
                model_training_run_id,
                trainRunConfig.train_run_id,
                user=x_consumer_username,
            )
            status_of_job = job_handler.status_of_job()
            if status_of_job == "Untraceable":
                detail_for_exception = {
                    "runId": str(model_training_run_id),
                    "status": "Could not get the status of the job",
                }
                raise HTTPException(status_code=404, detail=detail_for_exception)
            status_of_deletion = job_handler.delete_active_job(status_of_job, True)
            if status_of_deletion == "Failed":
                detail_for_exception = {
                    "runId": str(model_training_run_id),
                    "status": "Could not kill the job",
                }
                raise HTTPException(status_code=404, detail=detail_for_exception)
            result = {
                "runId": trainRunConfig.train_run_uuid,
                "status": "Killed the job successfully",
            }

        return result


@app.get(
    "/service/status",
    response_model=ServiceStatus,
    response_model_exclude_unset=True,
    tags=["Service"],
)
async def status():
    # TODO :: Add health checks like db connectivity
    return {"status": "OK"}


@app.get(
    "/service/healthcheck/gtg",
    response_model=ServiceStatus,
    response_model_exclude_unset=True,
    tags=["Service"],
)
async def healthcheck_gtg():
    return {"status": "OK"}


##-------------------------main------------------

setup_logging()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
