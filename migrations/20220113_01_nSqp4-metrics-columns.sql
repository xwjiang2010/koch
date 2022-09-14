-- Metrics Columns
-- depends: 20211208_01_hRulS-add-job-name-column-to-train-run-model-config
ALTER TABLE "kgsa-core-prediction-v1".train_run_model_config ADD COLUMN metrics jsonb;
ALTER TABLE "kgsa-core-prediction-v1".train_run_trial ADD COLUMN metrics jsonb;
