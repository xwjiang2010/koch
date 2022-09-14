from functools import partial
from os import environ
from typing import Dict, List, Union

import ray
from loguru import logger
from ray import tune

from prediction.app.prediction_commands import HyperparameterTuning
from prediction.app.prediction_handlers import PredictionHandler
from prediction.app.worker.training_handlers import DeepARModeler, TSAModeler
from prediction.src.static import PREDICTION_CONTEXT

CPU_COUNT = int(environ.get("RAY_TUNE_CPU_COUNT", "2"))


def runTrial(
    config: Dict,
    modeler: Union[TSAModeler, DeepARModeler],
    tuneCommand: HyperparameterTuning,
    target: str,
):
    modelName = tuneCommand.model_name
    model_run_id = tuneCommand.db_id
    ref = f"model_config_id:{model_run_id}-{modelName}"
    error = 1
    logger.info(
        f"*******{ref}--Running tune trial of {modelName} with configuration: {config}"
    )

    try:
        tuneCommand.model_params = config
        error, validation_error, metrics = modeler.handle(
            request=tuneCommand, target=target
        )
        logger.info(f"*******{ref}--Error of tune trial of {modelName}: {error}")

        report_results = {
            "mean_accuracy": 1 - error,
            "validation_accuracy": 1 - validation_error,
            "params": config,
            "model": modelName,
            "target": target,
            "done": True,
            "max_failures": 1,
        }

        report_results = {**report_results, **metrics}

        tune.report(**report_results)

    except Exception as e:
        logger.error(f"*****{ref}--Error in executing trial {tuneCommand}")
        logger.exception(e)


@ray.remote
class RayTuneExecutor(PredictionHandler):
    #  Read TUNE_QUEUE commands and generate commands for DATA_QUEUE
    #  Handles running tune algorithm
    #  As a subclass of tune.Trainable, Tune will create a Trainable object on a
    # separate process (using the Ray Actor API).
    def handle(
        self,
        request: HyperparameterTuning,
        modeler: Union[TSAModeler, DeepARModeler],
        tuneConfig: Dict,
        targets: List[str],
    ):
        # TODO
        # tune.run
        model = request.model_name
        ref = f"model_config_id:{request.db_id}-{model}"
        logger.info(
            f"************{ref}--START tuning execution tuneConfig: {tuneConfig}"
        )
        try:
            # TODO : Enable sync to s3 https://docs.ray.io/en/latest/tune/user-guide.html#distributed-checkpointing
            sync_config = tune.SyncConfig(
                sync_to_driver=False,
                sync_on_checkpoint=False,
                # upload_dir=PREDICTION_CONTEXT.S3_DURABLE_TRIAL_PATH,
            )
            cpu_count_by_model = {
                "ARIMA": 1,
                "HOLTWINTERS": 1,
                "PROPHET": 1,
            }
            cpu_per_trial = cpu_count_by_model.get(model, 1)
            train_experiments = [
                tune.Experiment(
                    f"{model}_{target_name}",
                    partial(
                        runTrial,
                        modeler=modeler,
                        tuneCommand=request,
                        target=target_name,
                    ),
                    config=tuneConfig,  # arima_config, # prophet_config, # holt_config,
                    max_failures=0,
                    sync_config=sync_config,
                    checkpoint_at_end=True,
                    resources_per_trial={"cpu": cpu_per_trial},
                    # sync_to_driver=False,
                    # sync_on_checkpoint=False,
                    # local_dir=f"/home/ray/ray_results/{target_name}",
                )
                for target_name in targets
            ]
            tune.run_experiments(
                train_experiments,
                concurrent=True,
                # resources_per_trial={"cpu": cpus},
                # resources_per_trial=tune.PlacementGroupFactory([{"CPU": cpus}]),
                raise_on_failed_trial=False,
                # loggers=[JsonLogger],
                # queue_trials=True,
            )
            # We can no longer rely on the built in methods like .get_best_trial()
            # This is due to the fact that we are using ray.tune.run_experiments
            # with multiple concurrent experiments. Since get_best_trial() is
            # aggregating the results of all experiments, it will return the
            # best trial accross all experiments, not per experiment. As a result,
            # we need to calculate the best trial per experiment on an individual
            # basis.
            analysis_df = tune.ExperimentAnalysis(
                train_experiments[0].checkpoint_dir
            ).dataframe()
            metrics_dfs = {}
            for experiment in train_experiments:
                try:
                    target = experiment.name
                    experiment_df = analysis_df[
                        analysis_df["logdir"].str.startswith(experiment.checkpoint_dir)
                    ]
                    # logger.debug(f"{target=}")
                    logger.debug(f"Result of tuning: {experiment_df}")
                    logger.debug(f"Tuning Stats: {experiment_df}")
                    metrics_dfs[target] = experiment_df
                except ValueError as e:
                    logger.error(f"Failed to find any results for {experiment.name}")
                    logger.exception(e)
            return metrics_dfs
        except Exception as e:
            # TODO :: handle exception by propogating to caller
            logger.error(f"*****{ref}--Tuning {model} threw exception.")
            logger.error(f"***{ref}--Model params causing error: {tuneConfig}")
            # logger.info(f"Target data: {target_df}")
            logger.exception(e)
            # raise e


class RayTuneConfigCreator(PredictionHandler):
    # Read MODEL_QUEUE commands and generate commands for TUNE_QUEUE
    # Handles parameter generation
    def handle(self, request: HyperparameterTuning):
        model = request.model_name
        params = request.model_params
        logger.info(f"# # Tune model: {model} params: {params}")
        if model == "ARIMA":
            modelConfig = {
                "p": tune.grid_search(params["p"]),
                "d": tune.grid_search(params["d"]),
                "q": tune.grid_search(params["q"]),
            }
        if model == "HOLTWINTERS":
            modelConfig = {
                "trend": tune.grid_search(params["trend"]),
                "damped": tune.grid_search(params["damped"]),
                "seasonal": tune.grid_search(params["seasonal"]),
            }
            if "seasonal_periods" in params:
                modelConfig["seasonal_periods"] = tune.grid_search([2, 3, 4, 6, 12])
        if model == "PROPHET":
            modelConfig = {
                "growth": tune.grid_search(params["growth"]),
                "n_changepoints": tune.grid_search(params["n_changepoints"]),
                "changepoint_range": tune.grid_search(params["changepoint_range"]),
                "changepoint_prior_scale": tune.grid_search(
                    params["changepoint_prior_scale"]
                ),
                "yearly_seasonality": tune.grid_search(params["yearly_seasonality"]),
                "weekly_seasonality": tune.grid_search(params["weekly_seasonality"]),
                "daily_seasonality": tune.grid_search(params["daily_seasonality"]),
                "seasonality_mode": tune.grid_search(params["seasonality_mode"]),
                "seasonality_prior_scale": tune.grid_search(
                    params["seasonality_prior_scale"]
                ),
                "holidays_prior_scale": tune.grid_search(
                    params["holidays_prior_scale"]
                ),
                "interval_width": tune.grid_search(params["interval_width"]),
                "uncertainty_samples": tune.grid_search(params["uncertainty_samples"]),
            }
            if "holidays" in params:
                modelConfig["holidays"] = tune.grid_search(params["holidays"])
            if "changepoints" in params:
                modelConfig["changepoints"] = tune.grid_search(params["changepoints"])

        # logger.info(f"# # Tune modelConfig: {pprint.pprint(modelConfig)}")

        logger.info(f"# # Tune modelConfig: {modelConfig}")
        return modelConfig
