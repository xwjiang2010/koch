# KGSA-Core-prediction

This repo provides the KGSA Core Prediction Service

[![pipeline status](https://kochsource.io/KBS_Analytics_Solutions/KBS/kgsa-core-apps/kgsa-core-prediction/badges/master/pipeline.svg)](https://kochsource.io/KBS_Analytics_Solutions/KBS/kgsa-core-apps/kgsa-core-prediction/-/commits/master)
[![coverage report](https://kochsource.io/KBS_Analytics_Solutions/KBS/kgsa-core-apps/kgsa-core-prediction/badges/master/coverage.svg)](https://kochsource.io/KBS_Analytics_Solutions/KBS/kgsa-core-apps/kgsa-core-prediction/-/commits/master)

[[_TOC_]]

## Branching Strategy
This project uses a modified [GitLab Flow](https://docs.gitlab.com/ee/topics/gitlab_flow.html), in short this is how it works:
1. Create a branch off of master
2. Commit Changes
3. Open a Merge Request Into Master
4. To release to QA open a Merge Request from `master` -> `release`
5. To release to Prod Tag a commit on the `release` branch (Ideally the HEAD commit)

**A Note about how the deployments work:**
- The following are protected:
  - branches: `master`, `release`
  - tags: `v*`
- Head of `master` is automatically deployed to Development Environment via CI/CD
- Head of `release` is automatically deployed to QA environment via CI/CD
- Tagged `release` commits are automatically deployed to Production
   - tags follow the [semantic versioning scheme](https://semver.org/) with a `v` prefix, for example `v1.3.0`
   - If you are making a major breaking change or introducing major functionality increment the major version
   - If you are making an additive change increment the minor version
   - If you are fixing a bug without introducing new functionality increment the patch version

## Getting Started
Initialize environment - grab some coffee this will take a while the first time (I blame torch) - especially if you have windows
```shell
$ make dev
```

This will install the necessary python environment and packages. Note this project uses Python `Poetry` to manage packages, this allows us to
lock down dependencies in a reliable manner. Instruction for adding packages and removing them can be found below.

### Managing Python Packages
- Adding packages: `poetry add PACKAGE_NAME==VERSION` for example - `poetry add pandas==1.1.3`
- Adding a development only package: `poetry add -D pytest`
- Removing packages: `poetry remove pandas`


### Running locally
#### Running with Poetry
Running outside of docker (natively) - this is the easiest and fastest way to test basic changes. Note this is still a WIP,
we still need to figure out a goo`d way to run the ray workers locally, which will likely involve minikube.
Nonetheless this will at least allow you to test if your python code is somewhat valid.

1. Set env vars
```shell
export RUN_LOCAL=1 # runs the application without a kubernetes job
export LOCAL_RAY_CLUSTER=1 # Runs a ray cluster locally vs in anyscale
export JSON_LOGS=0 # human readable logs
export DATABASE_URL=postgresql://postgres:local_db_password@localhost/postgres # local postgres db
export IAM_INSTANCE_PROFILE_ARN="arn:aws:iam::254486207130:instance-profile/ray-autoscaler-v1" # Set the policy used by the ray workers for s3 access
```
2. Start the service locally
`uvicorn --host 0.0.0.0 --port 8080 prediction.master_server:app`

Alternatively its also possible to use one of the other entry points like the scoring or training mediators (you'll need to set up the corresponding records in the db):
`python prediction/app/worker/scoring_mediator.py --run-id 603 --user devin --action score --log-level debug`

#### Running inside of Docker
Running inside of Docker is the most reliable way to ensure the application will run as expected when deployed.
Futhermore, running inside of minikube is the most complete way to test locally as it will not only run the container,
but also run within a kubernetes cluster. This allows training to be tested locally.

##### Docker Standalone Image
This will start up the application within docker, this option is somewhat limited as it will not have access to a cluster.
1. Build the image
   - `make build`
2. Start the container:
   - `make run`
3. Make changes
4. Stop the container (CTRL+C)
5. Return to Step 1 to rebuild and restart

##### Docker From Within MiniKube
This will start up the application within minikube, this will allow use to test cluster dependent functionality
1. Start MiniKube
   - `make start-minikube`
2. Setup Docker to Communicate with MiniKube
   - `eval $(minikube -p minikube docker-env)`
3. Build and push and deploy the image to minikube
   - `make deploy`
4. Make changes
5. Rebuild, push and redeploy
   - `make redeploy`
6. Repeat 4-5

## Making changes to the DB Schema
This project uses YoYo for db migrations. Here is a quick start:
### Set up the local db
`make init-db`

### Run existing migrations
`make migrate-db`

### Create a new migration
`poetry run yoyo new --sql -m "Some Meaningful Message (e.g. Adds column foo to table bar)"`

### Fix for broken migration error
raise InterpolationMissingOptionError(
configparser.InterpolationMissingOptionError: Bad value substitution: option 'database' in section 'DEFAULT' contains an interpolation key 'database_url' which is not a valid option name. Raw value: '%(DATABASE_URL)s'

run the below command to fix in your shell...

export DATABASE_URL=postgresql://postgres:local_db_password@localhost/postgres

### Apply migrations
`make migrate-db` -- alternatively you can use `poetry run yoyo apply -b`

**Note:** If you need to delete a local migration (never applied in production/dev/qa), you can stop the db docker container and re-apply the migrations
