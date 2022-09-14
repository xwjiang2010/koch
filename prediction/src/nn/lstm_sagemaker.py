import argparse
import json
import os
import pickle
import sys

import torch
from data import LSTM1_TestDataset, LSTM1_TrainDataset
from loguru import logger
from network import LSTM1, mape_loss
from torch.utils.data import DataLoader

logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.error(sys.path)

loss_dict = {
    "L1": torch.nn.L1Loss(),
    "MSE": torch.nn.MSELoss(),
    "SmoothL1": torch.nn.SmoothL1Loss(),
    "mape": mape_loss,
}


def train(args):
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # save environment variables to dict
    save_as_pickle(
        os.path.join(args.model_dir, "env_vars.pickle"),
        {k: v for k, v in os.environ.items() if k.startswith("SM_")},
    )

    # only for s3
    train_path = os.path.join(args.train_dir, "train.json")
    test_path = os.path.join(args.test_dir, "valid.json")
    #     create train and test loader
    train_data = LSTM1_TrainDataset(
        train_path, args.context_length, args.pred_length, normalized=args.normalized
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    if args.normalized:
        mean_std = (train_data.mean_, train_data.std_)
        logger.debug(mean_std)
        save_as_pickle(os.path.join(args.model_dir, "mean.pickle"), train_data.mean_)
        save_as_pickle(os.path.join(args.model_dir, "std.pickle"), train_data.std_)

    test_data = LSTM1_TestDataset(
        test_path,
        args.context_length,
        args.pred_length,
        *mean_std,
        normalized=args.normalized,
    )
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True)

    # define model, loss fn and optimizer
    model = LSTM1(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        output_dim=args.pred_length,
        seq_length=args.context_length,
        fully_connected=args.fully_connected,
        dropout=args.dropout,
        num_layers=args.num_layers,
        bias=args.bias,
    ).to(device)
    # check if model is on cuda
    logger.debug("Is model on cuda:{}".format(next(model.parameters()).is_cuda))
    loss_fn = loss_dict[args.loss_func]
    logger.info("Using loss function:{}".format(args.loss_func))
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    loss_history = {0: float("inf")}
    best_loss = float("inf")
    # Training loop
    for t in range(1, args.epochs):
        logger.debug("inside training loop")
        # checking status of GPU
        cuda_status(device, logger)

        model.train()
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            # Forward pass
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)
            if t % args.log_interval == 0:
                logger.info(
                    f"Epoch {t}|{(t/args.epochs):.2f} {args.loss_func}: {loss.item():.2e}"
                )
            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()
            # Backward pass
            loss.backward()
            # gradient clipping to avoid exploding gradients
            if args.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
                for p in model.parameters():
                    p.data.add_(-args.lr, p.grad.data)
            # Update parameters
            optimiser.step()

        logger.info(f"Train set: Train loss: {loss.item():.4f}")
        val_loss = test(model, test_loader, device, args)
        total_loss = val_loss + loss.item()
        logger.info(f"sum_Train_Test loss: {total_loss:.4f}")
        loss_history[t] = total_loss

        if total_loss < best_loss:
            best_loss = total_loss
            save_model(model, args.model_dir)
    save_as_pickle(os.path.join(args.model_dir, "loss.pickle"), loss_history)


def test(model, test_loader, device, args):
    model.eval()
    test_loss = 0
    loss_fn = loss_dict[args.test_loss_func]
    with torch.no_grad():
        i = 1
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            test_loss += loss_fn(y_pred, y_test).item()
            i = i + 1
    test_loss /= i
    logger.info(f"Test set: Average loss: {test_loss:.4f}")
    return test_loss


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(LSTM1())
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def save_as_pickle(path, var_to_pickle):
    logger.info(f"saving to ... {path}")
    with open(path, "wb") as handle:
        pickle.dump(var_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def cuda_status(device, logger):
    if device.type == "cuda":
        logger.info("GPU: {}".format(torch.cuda.get_device_name(0)))
        logger.info("GPU_ current device: {}".format(torch.cuda.current_device()))
        logger.info(
            "GPU Allocated: {} GB".format(
                round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
            )
        )
        logger.info(
            "GPU Cached: {} GB".format(
                round(torch.cuda.memory_cached(0) / 1024 ** 3, 1)
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for training (default: 1000)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=10000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=10,
        metavar="N",
        help="how many hidden dimensions for mlp input from lstm",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=7,
        metavar="N",
        help="sequence length of lstm",
    )
    parser.add_argument(
        "--pred-length",
        type=int,
        default=3,
        metavar="N",
        help="length of prediction vector",
    )
    parser.add_argument(
        "--normalized",
        type=bool,
        default=True,
        metavar="bool",
        help="whether data has to be normalized",
    )
    parser.add_argument(
        "--clip-value",
        type=float,
        default=None,
        metavar="1-5",
        help="gradient clipping vlaue for training",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=1,
        metavar="<1.0",
        help="dropout rate for multilayer lstm",
    )
    parser.add_argument(
        "--amsgrad",
        type=bool,
        default=True,
        metavar="bool",
        help="variant of adam optimizer. default True, turn off if needed",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        metavar="1-4",
        help="no of lstm layers stacked on top of each other",
    )
    parser.add_argument(
        "--bias",
        type=bool,
        default=True,
        metavar="bool",
        help="whether to add bias neural network weight multiplication",
    )
    parser.add_argument(
        "--loss-func",
        type=str,
        default="L1",
        metavar="L1, MSE, SmoothL1",
        help="type of loss function to use in training",
    )

    parser.add_argument(
        "--test-loss-func",
        type=str,
        default="L1",
        metavar="L1, MSE, SmoothL1",
        help="type of loss function to use for validation",
    )

    parser.add_argument(
        "--fully-connected",
        type=bool,
        default=True,
        metavar="bool",
        help=""""whether to connect all hidden layers or just last layer
                        to final mlp layer""",
    )

    # Container environment
    parser.add_argument(
        "--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"])
    )
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    #     parser.add_argument('--num-gpus', type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())
