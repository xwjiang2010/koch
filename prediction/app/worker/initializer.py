import configargparse

from prediction.log_config import setup_logging


def initialize(actions=["train", "predict"]) -> configargparse.Namespace:
    # Arguments are passed by the deployment file to the container
    parser = configargparse.ArgParser()
    parser.add("--run-id", type=str, required=True)
    parser.add("--user", type=str, required=True)
    parser.add("--action", type=str, choices=actions, required=True)
    parser.add(
        "--log-level",
        choices=["debug", "info", "warn", "warning", "error", "exception"],
        default="info",
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    return args


if __name__ == "__main__":
    args = initialize()
