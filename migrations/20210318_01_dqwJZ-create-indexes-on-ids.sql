-- create indexes on ids
-- depends: 20210216_01_2TkwM-add-feature-ranking-columns
CREATE index IF NOT EXISTS train_run_model_config_train_run_id_index ON "kgsa-core-prediction-v1".train_run_model_config (train_run_id DESC);
CREATE index IF NOT EXISTS train_run_trial_train_run_model_id_index ON "kgsa-core-prediction-v1".train_run_trial (train_run_model_id DESC);
