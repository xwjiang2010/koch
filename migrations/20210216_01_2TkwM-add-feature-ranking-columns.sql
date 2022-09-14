-- add feature ranking columns
-- depends: 20210205_01_Fssee-fresh-start-for-migrations
alter table "kgsa-core-prediction-v1".train_run_config
	add feature_columns JSONB;

alter table "kgsa-core-prediction-v1".train_run_trial
	add model_loss_function text;

alter table "kgsa-core-prediction-v1".train_run_trial
	add feature_columns JSONB;
