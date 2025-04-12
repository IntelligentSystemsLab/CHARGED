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
from joblib import Parallel, delayed
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

        def fit_predict(j):
            fit_series = train_valid_feat[:, j]
            model = AutoReg(fit_series, lags=self.lags)
            model_fitted = model.fit()
            pred_node = []
            for i in range(time_len):
                start = len(fit_series) + self.pred_len
                end = start
                pred = model_fitted.predict(start=start, end=end)
                pred_node.append(pred[-1])
            return pred_node

        preds = Parallel(n_jobs=10)(delayed(fit_predict)(j) for j in range(node))
        preds=np.array(preds).T
        return preds


class Arima:
    def __init__(self, pred_len):
        """
        Initialize the ARIMA model parameters.
        """
        self.pred_len = pred_len

    def predict(self, train_valid_feat, test_feat):
        time_len, node = test_feat.shape
        train_valid_feat = train_valid_feat[:-self.pred_len, :]
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=".*Maximum Likelihood optimization failed to converge.*")

        def fit_predict(j):
            fit_series = train_valid_feat[:, j]
            model = auto_arima(
                fit_series,
                start_p=0, max_p=2,
                start_q=0, max_q=2,
            )
            pred_node = []
            for i in range(time_len):
                pred = model.predict(n_periods=self.pred_len)
                pred_node.append(pred[-1])  # 只取最后一个值
            return pred_node

        preds = Parallel(n_jobs=10)(delayed(fit_predict)(j) for j in range(node))
        preds=np.array(preds).T
        return preds


class Fcnn(nn.Module):
    def __init__(self, n_fea, node, seq=12):  # input_dim = seq_length
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
    def __init__(self, seq, n_fea, node):
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
        return x  # [batch, node]


class SegRNN(nn.Module):

    def __init__(self, seq_len,pred_len,node,n_fea,seg_len=1,d_model=256,dropout=0.1):
        super(SegRNN, self).__init__()

        # get parameters
        self.seq_len = seq_len
        self.enc_in = n_fea
        self.d_model = d_model
        self.dropout = dropout
        self.pred_len = pred_len
        self.nodes = node

        self.seg_len = seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # building model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )

    def encoder(self, x):
        # b:batch_size c:channel_size s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last) # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        _, hn = self.rnn(x) # bc,n,d  1,bc,d

        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)) # bcm,1,d  1,bcm,d

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_last
        return y

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def forward(self,feat, extra_feat=None):
        x = feat.unsqueeze(-1)
        if extra_feat is not None:
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        B, node, seq_len, channel = x.shape
        x_enc = x.view(B * node, seq_len, channel)
        dec_out = self.forecast(x_enc)
        features = dec_out[:, -1, :]
        out = features.view(B, node)
        return out
