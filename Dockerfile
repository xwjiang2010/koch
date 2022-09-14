# syntax = docker/dockerfile:experimental

# `python-base` sets up all our shared environment variables
FROM kochsource.io:443/kbs_analytics_solutions/kbs/kgsa-core-apps/dependency_proxy/containers/python:3.8.6-slim as python-base

# python
ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.1.3 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    \
    # paths
    # this is where our requirements + virtual environment will live
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"


# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


# `builder-base` stage is used to build deps + create our virtual environment
FROM python-base as builder-base
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    # deps for installing poetry
    curl \
    # deps for building python deps
    build-essential \
    # deps for postgresql
    libpq-dev \
    python3-dev \
    procps \
    unzip

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSLcurl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# Install awscli for ray config_sync to s3
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# copy project requirement files here to ensure they will be cached.
WORKDIR $PYSETUP_PATH
COPY poetry.lock pyproject.toml ./

# install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
RUN poetry run pip install --upgrade pip
RUN poetry install
# --no-dev

COPY yoyo.ini .

############################### Local Development Image #######################################
# `development` image is used during development / testing locally, src files are mounted
# into the image which allows on the fly changes without rebuilding.
# FROM python-base as development
# ENV FLASK_ENV=development
# WORKDIR $PYSETUP_PATH
#
# # copy in our built poetry + venv
# COPY --from=builder-base $POETRY_HOME $POETRY_HOME
# COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH
# COPY --from=builder-base /usr/lib/x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/
# RUN apt-get update \
#     && apt-get install --no-install-recommends -y \
#     # deps for postgresql
#     libpq-dev
#
#
# # quicker install as runtime deps are already installed
# RUN pip install --upgrade pip
# RUN poetry install
#
# # will become mountpoint of our code
# WORKDIR /app
# EXPOSE 5000
#
# # Revisit making this async
# #CMD ["uvicorn", "--reload", "master_server:app", "--port", "8080"]
# CMD ["python", "-m", "prediction.master_server.py"]


############################### Production Image #######################################
# `production` image used for runtime
FROM python-base as production

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    # deps for postgresql
    libpq-dev \
    procps \
    # used by ray to copy tune logs
    rsync

COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH
COPY --from=builder-base /usr/lib/x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/
COPY --from=builder-base /usr/local/aws-cli/ /usr/local/aws-cli/

# This is a hack for the sake of ray head and workers
RUN echo 'PATH="$POETRY_HOME/bin:$VENV_PATH/bin:/usr/local/aws-cli/v2/current/bin:$PATH"' >> ~/.bashrc
RUN echo 'PYTHONPATH="/app/:$PYTHONPATH"' >> ~/.bashrc

COPY yoyo.ini /app/
COPY migrations /app/migrations/
COPY prediction /app/prediction/
WORKDIR /app
CMD ["ddtrace-run", "uvicorn", "--host", "0.0.0.0", "--port", "5000", "prediction.master_server:app"]

# ############################### Test Image #######################################
# FROM production as test
# WORKDIR $PYSETUP_PATH
# COPY --from=builder-base $POETRY_HOME $POETRY_HOME
# #COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH
#
# # install dev dependencies for testing
# RUN poetry install
#
# WORKDIR /app
# COPY conftest.py /app/
# COPY tests /app/tests
# CMD ["pytest", "--cov=src", "tests"]
