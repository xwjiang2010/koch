#!/bin/sh
# Change port to local port.
curl --location --request POST 'http://0.0.0.0:5000/training' \
--header 'Authorization: Basic {Credentials}=' \
--header 'X-Consumer-Username: test-local-user-name' \
--header 'Content-Type: application/json' \
--data-raw '{
    "master": {
        "exec_environment": "Parallel",
        "data_storage_type": "s3",
        "data_file_type": "csv",
        "resources": {
            "max_parallel_pods": 1
        }
    },
    "models": {
        "ARIMA": {
        "model_name": "ARIMA",
        "model_config": {
            "hyperparameters": {
                "disp": 0
            }
        }
    }
    },
    "training": {
        "train_task_type":
        "Tuning","train_name":
        "Tuning-Molex-data-postman-test",
        "target_data_location": "s3://prediction-services/data/test_single_target.csv",
        "score_data_location": "s3://prediction-services/score/",
        "train_data_end_dtm": "1990-01-31","test_data_end_dtm": "1990-02-03",
        "validation_data_end_dtm": "1990-03-02",
        "model_names": ["ENSEMBLE"]
    }
}'
