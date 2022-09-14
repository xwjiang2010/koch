-- Update for user run tracking
-- depends: 20210329_01_4i0kZ-adding-a-new-column-to-the-train-run-config-table-to-keep-track-of-number-of-pods-generated
alter table "kgsa-core-prediction-v1".train_run_config
	add username text;

alter table "kgsa-core-prediction-v1".score_run_config
	add username text;

create index if not exists train_run_config_username_index
	on "kgsa-core-prediction-v1".train_run_config (username);

create index if not exists score_run_config_username_index
	on "kgsa-core-prediction-v1".score_run_config (username);