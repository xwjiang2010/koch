-- Add job name column to train run model config
-- depends: 20211124_01_l1oLe-add-prediction-steps-to-train-run-config
alter table "kgsa-core-prediction-v1".train_run_model_config add model_job_name text;
