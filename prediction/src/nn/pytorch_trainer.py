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

from prediction.src.static import PREDICTION_CONTEXT

loss_dict = {
    "L1": torch.nn.L1Loss(),
    "MSE": torch.nn.MSELoss(),
    "SmoothL1": torch.nn.SmoothL1Loss(),
    "mape": mape_loss,
}


def train(args, model, trainData, testData):
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    loss_fn = loss_dict[args.loss_func]
    model = model.to(device)

    ## Training loop
    for epoch in range(args.epochs):
        logger.debug("inside training loop")
        running_loss = 0.0
        for X_train, y_train in trainData:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            X_train, y_train = X_train.to(device), y_train.to(device)

            # Step 3. Run our forward pass.
            y_pred = model(X_train)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_fn(y_pred, y_train)
            running_loss += loss

            ## Step 5: Backward pass
            loss.backward()

            # Step 6: gradient clipping to avoid exploding gradients
            if args.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
                for p in model.parameters():
                    p.data.add_(-args.lr, p.grad.data)

            # Step 7: Update parameters
            optimiser.step()
