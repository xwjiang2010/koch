-- Add prediction steps to train_run_config
-- depends: 20210811_01_EiY2G-Update-for-user-run-tracking
alter table "kgsa-core-prediction-v1".train_run_config add prediction_steps int;