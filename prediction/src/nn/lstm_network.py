import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F


## defining the model
class LSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        seq_length,
        bias=True,
        batch_first=True,
        dropout=0,
        num_layers=1,
        fully_connected=False,
    ):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.num_layers = num_layers
        self.fully_connected = fully_connected

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=False,
        )

        # Define the output layer
        if self.fully_connected:
            self.linear = nn.Linear(self.seq_length * self.hidden_dim, self.output_dim)
        else:
            self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, X):
        # Forward pass through LSTM layer
        lstm_out, _ = self.lstm(X.view(-1, self.seq_length, self.input_dim))
        if self.fully_connected:
            linear_in = lstm_out.contiguous().view(
                -1, self.seq_length * self.hidden_dim
            )
        else:
            linear_in = lstm_out[:, -1, :].contiguous().view(-1, self.hidden_dim)

        y_pred = self.linear(linear_in)
        return y_pred
