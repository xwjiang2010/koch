--
-- depends: 20220110_01_MTN5D-add-feature-mapping-to-train-config
alter table "kgsa-core-prediction-v1".score_run_config add target_feature_mapping_location text;
