PYTHON_VERSION = 3.8.6

.PHONY: all dev pyenv install-poetry build-dev run-dev build run push test deploy ci-test migrate-db


DB_PASSWORD ?= local_db_password
POSTGRES_PASSWORD ?= ${DB_PASSWORD}

DB_USERNAME ?= postgres
POSTGRES_USER ?= ${DB_USERNAME}

DB_HOST ?= localhost
POSTGRES_HOST ?= ${DB_HOST}

POSTGRES_DB ?= postgres

DATABASE_URL ?= postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}/${POSTGRES_DB}

all:

# In test stage always use the constituant parts to define the url, else allow it to use an existing DATABASE_URL
pyenv:
	set -e;
	pyenv install -s ${PYTHON_VERSION}; \
	pyenv virtualenv -f ${PYTHON_VERSION} prediction-service;

install-poetry: pyenv
	@curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

install-docker:
	@set -e; \
	sudo apt install docker.io; \
	sudo usermod -aG docker $$USER; \

dev: install-poetry install-docker
	@set -e; \
	sudo apt install postgresql postgresql-contrib libpq-dev; \
	pip install --upgrade pip; \
	poetry install; \
	poetry run pre-commit install; \
	echo initialized; \

clean-db:
	docker stack rm postgres

init-db:
	@docker pull postgres; \
	docker swarm init; \
	docker stack deploy -c db_stack.yaml postgres;

migrate-db:
	@set -e; \
	echo $(DATABASE_URL); \
	poetry run yoyo apply -b -vvv; \

migration:
	@set -e; \
	poetry run yoyo new --sql

build-dev:
	@docker build . --target development -t kgsa-core-prediction-dev

build:
	@set -e; \
	set -x; \
	docker build . --target production -t kgsa-core-prediction;

helm-postgres:
	@helm repo add bitnami https://charts.bitnami.com/bitnami; \
	helm upgrade --install postgres --set postgresqlPassword=local_db_password,postgresqlDatabase=postgres bitnami/postgresql

run:
	@uvicorn --host 0.0.0.0 --port 5000 prediction.master_server:app

test:
	@poetry run pytest -m "not ray" --cov=prediction tests

ci-test:
	@docker run --rm -it kgsa-core-prediction-test
