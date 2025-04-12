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
import torch.nn.functional as F
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

    def __init__(self, seq_len,pred_len,n_fea,seg_len=1,d_model=256,dropout=0.1):
        super(SegRNN, self).__init__()

        # get parameters
        self.seq_len = seq_len
        self.enc_in = n_fea
        self.d_model = d_model
        self.dropout = dropout
        self.pred_len = pred_len

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

    def forward(self,feat, extra_feat=None, chunk_size=256):
        x = feat.unsqueeze(-1)
        if extra_feat is not None:
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        B, node, seq_len, channel = x.shape
        x_enc = x.view(B * node, seq_len, channel)
        outputs = []
        for i in range(0, x_enc.shape[0], chunk_size):
            x_chunk = x_enc[i: i + chunk_size]
            dec_out_chunk = self.forecast(x_chunk)
            features_chunk = dec_out_chunk[:, -1, :]
            outputs.append(features_chunk)
        out = torch.cat(outputs, dim=0)
        out = out.view(B, node)
        return out

class FreTS(nn.Module):
    def __init__(self, seq_len,pred_len,n_fea,embed_size = 128,hidden_size = 256,sparsity_threshold = 0.01,scale = 0.02):
        super(FreTS, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.pre_length = pred_len
        self.feature_size = n_fea
        self.seq_length = seq_len
        self.sparsity_threshold = sparsity_threshold
        self.scale = scale
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, feat, extra_feat=None):
        x = feat.unsqueeze(-1)
        if extra_feat is not None:
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        B_ori, node, seq_len, channel = x.shape
        x_enc = x.view(B_ori * node, seq_len, channel)

        # x: [Batch, Input length, Channel]
        B, T, N = x_enc.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        bias = x
        # [B, N, T, D]
        # if self.channel_independence == '1':
        #     x = self.MLP_channel(x, B, N, T)
        x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x