--
-- depends: 20211208_01_hRulS-add-job-name-column-to-train-run-model-config
alter table "kgsa-core-prediction-v1".train_run_config add target_feature_mapping_location text;
