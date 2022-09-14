-- Adding a new column to the train_run_config table to keep track of number of pods generated
-- depends: 20210318_01_dqwJZ-create-indexes-on-ids
alter table "kgsa-core-prediction-v1".train_run_config
	add jobs_count int;
