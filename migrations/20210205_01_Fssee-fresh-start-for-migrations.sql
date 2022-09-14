-- Fresh Start For Migrations
-- depends:

create schema if not exists "kgsa-core-prediction-v1";

create table if not exists "kgsa-core-prediction-v1".train_run_config
(
	train_run_id serial not null
		constraint train_run_config_pk
			primary key,
	train_name text not null,
	train_description text,
	train_job_type text not null,
	target_data_location text not null,
	train_data_file_type text not null,
	train_data_storage_type text not null,
	feature_data_location text,
	model_artifacts_location text not null,
	score_data_location text,
	train_status text,
	train_start_ts timestamp,
	train_end_ts timestamp,
	model_version text,
	data_version text,
	update_ts timestamp default CURRENT_TIMESTAMP,
	train_run_uuid uuid not null,
	train_exec_environment text,
	train_task_type text,
	train_data_end_dtm timestamp,
	test_data_end_dtm timestamp,
	validation_data_end_dtm timestamp,
	model_type text
);

alter table "kgsa-core-prediction-v1".train_run_config owner to postgres;

create unique index if not exists train_run_config_train_run_id_uindex
	on "kgsa-core-prediction-v1".train_run_config (train_run_id);

create unique index if not exists train_run_config_train_run_uuid_uindex
	on "kgsa-core-prediction-v1".train_run_config (train_run_uuid);

create index if not exists train_run_config_train_run_uuid_index
	on "kgsa-core-prediction-v1".train_run_config (train_run_uuid);

create table if not exists "kgsa-core-prediction-v1".train_run_model_config
(
	train_run_model_id serial not null
		constraint train_run_models_config_pk
			primary key,
	train_run_id serial not null
		constraint train_run_model_config_train_run_config_train_run_id_fk
			references "kgsa-core-prediction-v1".train_run_config,
	model_name text not null,
	model_config jsonb not null,
	model_loss_function text,
	model_time_interval text,
	train_start_ts timestamp,
	train_end_ts timestamp,
	tune_algorithm text default 'grid-search'::text,
	model_artifacts_location text,
	update_ts timestamp default CURRENT_TIMESTAMP not null
);

alter table "kgsa-core-prediction-v1".train_run_model_config owner to postgres;

create unique index if not exists train_run_models_config_train_run_model_id_uindex
	on "kgsa-core-prediction-v1".train_run_model_config (train_run_model_id);

create table if not exists "kgsa-core-prediction-v1".train_run_trial
(
	train_run_trial_id serial not null
		constraint train_run_trial_pk
			primary key,
	train_run_model_id serial not null
		constraint train_run_trial_train_run_model_config_train_run_model_id_fk
			references "kgsa-core-prediction-v1".train_run_model_config,
	train_model_config jsonb not null,
	test_score real,
	trial_status text,
	validation_score real,
	trial_start_ts timestamp,
	trial_end_ts timestamp,
	model_trial_location text,
	update_ts timestamp default CURRENT_DATE not null,
	train_score real,
	target_column_name text
);

alter table "kgsa-core-prediction-v1".train_run_trial owner to postgres;

create unique index if not exists train_run_trial_train_run_model_stat_id_uindex
	on "kgsa-core-prediction-v1".train_run_trial (train_run_trial_id);

create table if not exists "kgsa-core-prediction-v1".model_parameter
(
	param_id serial not null,
	model_name text not null,
	model_type text not null,
	model_id serial not null,
	param_category text not null,
	param_name text not null,
	param_type text not null,
	param_required boolean not null,
	param_default_value text,
	update_ts timestamp default CURRENT_DATE not null,
	constraint model_parameter_pk
		primary key (param_id, model_id)
);

alter table "kgsa-core-prediction-v1".model_parameter owner to postgres;

create table if not exists "kgsa-core-prediction-v1".score_run_config
(
	score_run_id serial not null
		constraint score_run_config_pk
			primary key,
	score_run_uuid uuid not null,
	score_exec_environment text,
	score_data_storage_type text,
	score_data_file_type text,
	score_name text,
	score_description text,
	train_run_id integer
		constraint score_run_config_train_run_config_train_run_id_fk
			references "kgsa-core-prediction-v1".train_run_config,
	target_data_location text,
	feature_data_location text,
	score_data_location text,
	data_version text,
	score_status text,
	score_start_ts timestamp,
	score_end_ts timestamp,
	update_ts timestamp default now() not null,
	model_type text
);

alter table "kgsa-core-prediction-v1".score_run_config owner to postgres;

create unique index if not exists score_run_config_score_run_id_uindex
	on "kgsa-core-prediction-v1".score_run_config (score_run_id);

create unique index if not exists score_run_config_score_run_uuid_uindex
	on "kgsa-core-prediction-v1".score_run_config (score_run_uuid);

create table if not exists "kgsa-core-prediction-v1".score_model_summary
(
	score_model_id serial not null
		constraint score_model_summary_pk
			primary key,
	score_run_id serial not null
		constraint score_model_summary_score_run_config_score_run_id_fk
			references "kgsa-core-prediction-v1".score_run_config,
	model_name text,
	model_config jsonb,
	score_loss_function text,
	prediction_steps integer,
	prediction_count integer,
	score_start_ts timestamp,
	score_end_ts timestamp,
	target_column_name text not null,
	score_trial_status text,
	forecast_ts timestamp,
	forecasted_quantity jsonb,
	forecasted_interval jsonb,
	actual_quantity jsonb,
	update_ts timestamp default now() not null,
	score_time_interval text
);

alter table "kgsa-core-prediction-v1".score_model_summary owner to postgres;
