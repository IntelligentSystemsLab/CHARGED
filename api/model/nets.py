# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/11 2:14
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/11 2:14

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import warnings


class Lo:
    def __init__(self, pre_len):
        self.pred_len = pre_len

    def predict(self, train_valid_feat, test_feat):
        """
        Use the latest observed value as the prediction for the next time step.
        """
        time_len, node = test_feat.shape
        preds = np.zeros((time_len, node))

        for j in range(node):
            for i in range(time_len):
                if i < self.pred_len:
                    preds[i, j] = train_valid_feat[-self.pred_len + i, j]
                else:
                    preds[i, j] = test_feat[i - self.pred_len, j]

        return preds


class Ar:
    def __init__(self, pred_len, lags=1):

        self.pred_len = pred_len
        self.lags = lags

    def predict(self, train_valid_feat, test_feat):
        """
        Perform predictions using the AR model.
        """
        time_len, node = test_feat.shape
        train_valid_feat = train_valid_feat[:-self.pred_len, :]
        preds = np.zeros((time_len, node))

        for j in range(node):  # Train and predict for each node
            fit_series = train_valid_feat[:, j]

            # Train AR model on each node
            model = AutoReg(fit_series, lags=self.lags)
            model_fitted = model.fit()
            for i in range(time_len):
                start = len(fit_series) + self.pred_len
                end = start
                pred = model_fitted.predict(start=start, end=end)
                preds[i, j] = pred[0]  # Ensure single value is assigned

        return preds


class Arima:
    def __init__(self, pred_len, p=1, d=1, q=1):
        """
        Initialize the ARIMA model parameters.
        """
        self.pred_len = pred_len
        self.p = p
        self.d = d
        self.q = q

    def predict(self, train_valid_feat, test_feat):
        time_len, node = test_feat.shape
        train_valid_feat = train_valid_feat[:-self.pred_len, :]
        preds = np.zeros((time_len, node))
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=".*Maximum Likelihood optimization failed to converge.*")
        for j in range(node):  # Train and predict for each node
            fit_series = train_valid_feat[:, j]

            # Train ARIMA model on each node
            model=auto_arima(fit_series)

            # Predict using the ARIMA model
            for i in range(time_len):
                pred = model.predict(n_periods=self.pred_len)
                preds[i, j] = pred[-1]  # Ensure single value is assigned

        return preds


class Fcnn(nn.Module):
    def __init__(self, n_fea, node=247, seq=12):  # input_dim = seq_length
        super(Fcnn, self).__init__()
        self.num_feat = n_fea
        self.seq = seq
        self.nodes = node
        self.linear = nn.Linear(seq * n_fea, 1)

    def forward(self, feat, extra_feat=None):
        x = feat  # batch, nodes,seq for region or batch, seq for nodes
        if extra_feat is not None:
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
            assert x.shape[
                       -1] == self.num_feat, f"Number of features ({x.shape[-1]}) does not match n_fea ({self.num_feat})."
        x = x.view(-1, self.nodes, self.seq * self.num_feat)
        x = self.linear(x)
        x = torch.squeeze(x)
        return x


class Lstm(nn.Module):
    def __init__(self, seq, n_fea, node=331):
        super(Lstm, self).__init__()
        self.num_feat = n_fea
        self.nodes = node
        self.seq_len = seq
        self.encoder = nn.Linear(n_fea, 1)  # input.shape: [batch, channel, width, height]
        self.lstm_hidden_dim = 16
        self.lstm = nn.LSTM(input_size=n_fea, hidden_size=self.lstm_hidden_dim, num_layers=2,
                            batch_first=True)
        self.linear = nn.Linear(seq * self.lstm_hidden_dim, 1)

    def forward(self, feat, extra_feat=None):  # feat.shape = [batch, node, seq]
        x = feat.unsqueeze(-1)
        if extra_feat is not None:
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        assert x.shape[
                   -1] == self.num_feat, f"Number of features ({x.shape[-1]}) does not match n_fea ({self.num_feat})."

        bs = x.shape[0]
        x = x.view(bs * self.nodes, self.seq_len, self.num_feat)

        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch_size * node, seq_len, lstm_hidden_dim]
        lstm_out = lstm_out.reshape(bs, self.nodes, self.seq_len * self.lstm_hidden_dim)
        x = self.linear(lstm_out)
        x = torch.squeeze(x)
        return x
