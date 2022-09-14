import os
from contextlib import contextmanager

import ray
from anyscale import AnyscaleSDK
from anyscale.sdk.anyscale_client.models import (
    CloudsQuery,
    ClusterComputeConfig,
    ComputeNodeType,
    TextQuery,
    WorkerNodeType,
)
from loguru import logger

from prediction.app.prediction_commands import IAM_INSTANCE_PROFILE_ARN

CLUSTER_ENV = os.environ.get("CLUSTER_ENV", "prediction-service:49")
LOCAL_RAY_CLUSTER = bool(int(os.environ.get("LOCAL_RAY_CLUSTER", "0")))
CLUSTER_COMPUTE_TYPE = os.environ.get("CLUSTER_COMPUTE_TYPE", "xlarge_instances")
IAM_INSTANCE_PROFILE_ARN = os.environ.get("IAM_INSTANCE_PROFILE_ARN", "")


# TODO :: Fetch Valid Configurations Per User Group (Future feature)
def get_cluster_compute_config(cluster_compute_type: str) -> ClusterComputeConfig:
    asdk = AnyscaleSDK()
    clouds = asdk.search_clouds(
        CloudsQuery(name=TextQuery(contains="koch-anyscale-app-integration"))
    )
    cloud_id = clouds.results[0].id
    aws_node_options = (
        {"IamInstanceProfile": {"Arn": IAM_INSTANCE_PROFILE_ARN}}
        if IAM_INSTANCE_PROFILE_ARN
        else {}
    )

    logger.info(f"Using cluster_compute_type: {cluster_compute_type}")
    cluster_computes = {
        "xlarge_instances": ClusterComputeConfig(
            cloud_id=cloud_id,
            region="us-east-1",
            head_node_type=ComputeNodeType(
                name="Head", instance_type="t3.2xlarge", resources={"cpu": 0}
            ),
            worker_node_types=[
                WorkerNodeType(
                    name="worker-node-group-1",
                    instance_type="m5.24xlarge",
                    use_spot=False,
                    max_workers=10,
                )
            ],
            aws=aws_node_options,
        ),
        "large_instances": ClusterComputeConfig(
            cloud_id=cloud_id,
            region="us-east-1",
            head_node_type=ComputeNodeType(name="Head", instance_type="t3.2xlarge"),
            worker_node_types=[
                WorkerNodeType(
                    name="worker-node-group-1",
                    instance_type="m5.9xlarge",
                    use_spot=False,
                    max_workers=10,
                )
            ],
            aws=aws_node_options,
        ),
        "medium_instances": ClusterComputeConfig(
            cloud_id=cloud_id,
            region="us-east-1",
            head_node_type=ComputeNodeType(name="Head", instance_type="m5.8xlarge"),
            worker_node_types=[
                WorkerNodeType(
                    name="worker-node-group-1",
                    instance_type="c5.4xlarge",
                    min_workers=4,
                    max_workers=4,
                    use_spot=False,
                )
            ],
            aws=aws_node_options,
        ),
        "small_instances": ClusterComputeConfig(
            cloud_id=cloud_id,
            region="us-east-1",
            head_node_type=ComputeNodeType(name="Head", instance_type="m5.8xlarge"),
            worker_node_types=[
                WorkerNodeType(
                    name="worker-node-group-1",
                    instance_type="m5.xlarge",
                    min_workers=4,
                    max_workers=4,
                    use_spot=False,
                )
            ],
            aws=aws_node_options,
        ),
    }
    return cluster_computes[cluster_compute_type]


@contextmanager
def ray_cluster(action, run_id, index=0):
    cluster_name = f"{action}-{run_id}-{index}"

    with logger.contextualize(cluster_name=cluster_name):
        if not LOCAL_RAY_CLUSTER:
            cluster_compute = get_cluster_compute_config(CLUSTER_COMPUTE_TYPE)
            cluster = f"anyscale://{cluster_name}"
            logger.info(f"Starting anyscale cluster, address: {cluster}")
            # This is a workaround for the ray connection issue
            # We need to call ray init once to start the cluster,
            # and a second time to connect to the cluster
            try:
                with ray.init(
                    cluster,
                    project_name="prediction-service",
                    cluster_env=CLUSTER_ENV,
                    cluster_compute=cluster_compute.to_dict(),
                    autosuspend="10m",
                    runtime_env={"working_dir": "./", "excludes": ["tests/", "./.*"]},
                ) as client:
                    client.disconnect()
            except Exception as e:
                logger.error(f"Error in ray_cluster startup: {e}")
            with ray.init(
                cluster,
                project_name="prediction-service",
                cluster_env=CLUSTER_ENV,
                cluster_compute=cluster_compute.to_dict(),
                log_to_driver=False,
                autosuspend="10m",
                runtime_env={"working_dir": "./", "excludes": ["tests/", "./.*"]},
            ) as client:
                try:
                    yield client
                finally:
                    client.disconnect()
        else:
            logger.info("Starting ray locally")
            yield ray.init(include_dashboard=False, local_mode=True)
