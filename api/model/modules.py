# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/11 2:14
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/11 2:14

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from joblib import Parallel, delayed
from pmdarima import auto_arima
from statsmodels.tsa.ar_model import AutoReg
from einops import rearrange

from api.model.layers import (
    ModernTCN_RevIN,
    ModernTCN_Stage,
    ModernTCN_Flatten_Head,
    AttentionLayer,
    FullAttention,
    MultiPatchFormer_Encoder,
    ConvTimeNet_ConvEncoder,
    ConvTimeNet_get_activation_fn,
)

"""
Module implementing a variety of time-series forecasting algorithms and neural models.

Classes:
    Lo: Last-observation baseline predictor.
    Ar: Autoregressive model predictor.
    Arima: ARIMA model predictor with automated order selection.
    Fcnn: Fully-connected neural predictor.
    Lstm: LSTM-based sequence predictor.
    SegRNN: Segmented RNN forecasting model.
    FreTS: Frequency-domain temporal-spatial model.
    ModernTCN: Temporal convolutional network with Reversible IN and re-parameterization.
    MultiPatchFormer: Multi-patch transformer encoder-based forecaster.
    ConvTimeNet: Convolutional time-series encoder.

Each class provides a `predict` or `forward` method for inference.
"""


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
        preds = np.array(preds).T
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
        preds = np.array(preds).T
        return preds


class Fcnn(nn.Module):
    def __init__(self, n_fea, node, seq=12):  # input_dim = seq_length
        super(Fcnn, self).__init__()
        self.num_feat = n_fea
        self.seq = seq
        self.nodes = node
        self.linear = nn.Linear(seq * n_fea, 1)

    def forward(self, feat, extra_feat=None):
        x = feat
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

    def __init__(self, seq_len, pred_len, n_fea, seg_len=1, d_model=256, dropout=0.1):
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
        self.final_linear = nn.Linear(self.enc_in, 1)

    def encoder(self, x):
        # b:batch_size c:channel_size s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last)  # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        _, hn = self.rnn(x)  # bc,n,d  1,bc,d

        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))  # bcm,1,d  1,bcm,d

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_last
        y = self.final_linear(y)
        return y

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def forward(self, feat, extra_feat=None):
        x = feat.unsqueeze(-1)
        if extra_feat is not None:
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        B_ori, node, seq_len, channel = x.shape
        x_enc = x.view(B_ori * node, seq_len, channel)
        outputs = []
        current_chunk_size = self.chunk_size
        start_idx = 0
        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx: start_idx + current_chunk_size]
                dec_out_chunk = self.forecast(x_chunk)
                features_chunk = dec_out_chunk[:, -1, :]
                outputs.append(features_chunk)
                start_idx += current_chunk_size
            except RuntimeError as e:
                if "CUDA" in str(e):
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e
        out = torch.cat(outputs, dim=0)
        out = out.view(B_ori, node)
        return out


class FreTS(nn.Module):
    def __init__(self, seq_len, pred_len, n_fea, embed_size=128, hidden_size=256, sparsity_threshold=0.01, scale=0.02):
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
        self.final_linear = nn.Linear(self.feature_size, 1)

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
        outputs = []
        current_chunk_size = self.chunk_size
        start_idx = 0
        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx: start_idx + current_chunk_size]
                B, T, N = x_chunk.shape
                x_chunk = self.tokenEmb(x_chunk)
                bias = x_chunk
                x_chunk = self.MLP_channel(x_chunk, B, N, T)
                x_chunk = self.MLP_temporal(x_chunk, B, N, T)
                x_chunk = x_chunk + bias
                x_chunk = self.fc(x_chunk.reshape(B, N, -1)).permute(0, 2, 1)
                x_chunk = self.final_linear(x_chunk)
                x_chunk = x_chunk.squeeze(-1)
                outputs.append(x_chunk)
                start_idx += current_chunk_size
            except RuntimeError as e:
                if "CUDA" in str(e):
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e
        out = torch.cat(outputs, dim=0)
        out = out.view(B_ori, node)
        return out


class ModernTCN(nn.Module):
    def __init__(self, n_fea, seq_len, pred_len, patch_size=16, patch_stride=8, downsample_ratio=2, ffn_ratio=2,
                 num_blocks=[1, 1, 1, 1],
                 large_size=[31, 29, 27, 13], small_size=[5, 5, 5, 5], dims=[256, 256, 256, 256],
                 dw_dims=[256, 256, 256, 256],
                 small_kernel_merged=False, backbone_dropout=0.1, head_dropout=0.1, use_multi_scale=True, revin=True,
                 affine=True,
                 subtract_last=False, individual=False, ):

        super(ModernTCN, self).__init__()
        c_in = n_fea
        target_window = pred_len
        # RevIN
        self.revin = revin
        if self.revin: self.revin_layer = ModernTCN_RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # stem layer & down sampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)

        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        # backbone
        self.num_stage = len(num_blocks)
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = ModernTCN_Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx],
                                    dmodel=dims[stage_idx],
                                    dw_model=dw_dims[stage_idx], nvars=n_fea, small_kernel_merged=small_kernel_merged,
                                    drop=backbone_dropout)
            self.stages.append(layer)

        # head
        patch_num = seq_len // patch_stride
        self.n_vars = c_in
        self.individual = individual
        d_model = dims[self.num_stage - 1]

        if use_multi_scale:
            self.head_nf = d_model * patch_num
            self.head = ModernTCN_Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                               head_dropout=head_dropout)
        else:

            if patch_num % pow(downsample_ratio, (self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsample_ratio, (self.num_stage - 1))
            else:
                self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1)) + 1)
            self.head = ModernTCN_Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                               head_dropout=head_dropout)

        self.final_linear = nn.Linear(n_fea, 1)

    def forward_feature(self, x):

        B, M, L = x.shape
        x = x.unsqueeze(-2)

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x

    def forward(self, feat, extra_feat=None):
        x = feat.unsqueeze(-1)
        if extra_feat is not None:
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        B_ori, node, seq_len, channel = x.shape
        x_enc = x.view(B_ori * node, seq_len, channel)
        outputs = []
        current_chunk_size = self.chunk_size
        start_idx = 0
        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx: start_idx + current_chunk_size]
                if self.revin:
                    x_chunk = self.revin_layer(x_chunk, 'norm')
                    x_chunk = x_chunk.permute(0, 2, 1)
                x_chunk = self.forward_feature(x_chunk)
                x_chunk = self.head(x_chunk)
                # de-instance norm
                if self.revin:
                    x_chunk = x_chunk.permute(0, 2, 1)
                    x_chunk = self.revin_layer(x_chunk, 'denorm')
                x_chunk = self.final_linear(x_chunk)
                x_chunk = x_chunk.squeeze(-1)
                outputs.append(x_chunk)
                start_idx += current_chunk_size
            except RuntimeError as e:
                if "CUDA" in str(e):
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e
        out = torch.cat(outputs, dim=0)
        out = out.view(B_ori, node)
        return out

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


class MultiPatchFormer(nn.Module):
    def __init__(self, seq_len, pred_len, n_fea, e_layers=2, d_model=256, d_ff=1024, n_heads=8, dropout=0.1):
        super(MultiPatchFormer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_channel = n_fea
        self.N = e_layers
        # Embedding
        self.d_model = d_model
        self.d_hidden = d_ff
        self.n_heads = n_heads
        self.mask = True
        self.dropout = dropout

        self.stride1 = 3
        self.patch_len1 = 4
        self.stride2 = 4
        self.patch_len2 = 6
        self.stride3 = 3
        self.patch_len3 = 8
        self.stride4 = 2
        self.patch_len4 = 10
        self.patch_num1 = int((self.seq_len - self.patch_len2) // self.stride2) + 2
        self.padding_patch_layer1 = nn.ReplicationPad1d((0, self.stride1))
        self.padding_patch_layer2 = nn.ReplicationPad1d((0, self.stride2))
        self.padding_patch_layer3 = nn.ReplicationPad1d((0, self.stride3))
        self.padding_patch_layer4 = nn.ReplicationPad1d((0, self.stride4))

        self.shared_MHA = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(mask_flag=self.mask),
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                )
                for _ in range(self.N)
            ]
        )

        self.shared_MHA_ch = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(mask_flag=self.mask),
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                )
                for _ in range(self.N)
            ]
        )

        self.encoder_list = nn.ModuleList(
            [
                MultiPatchFormer_Encoder(
                    d_model=self.d_model,
                    mha=self.shared_MHA[ll],
                    d_hidden=self.d_hidden,
                    dropout=self.dropout,
                    channel_wise=False,
                )
                for ll in range(self.N)
            ]
        )

        self.encoder_list_ch = nn.ModuleList(
            [
                MultiPatchFormer_Encoder(
                    d_model=self.d_model,
                    mha=self.shared_MHA_ch[0],
                    d_hidden=self.d_hidden,
                    dropout=self.dropout,
                    channel_wise=True,
                )
                for ll in range(self.N)
            ]
        )

        pe = torch.zeros(self.patch_num1, self.d_model)
        for pos in range(self.patch_num1):
            for i in range(0, self.d_model, 2):
                wavelength = 10000 ** ((2 * i) / self.d_model)
                pe[pos, i] = math.sin(pos / wavelength)
                pe[pos, i + 1] = math.cos(pos / wavelength)
        pe = pe.unsqueeze(0)  # add a batch dimention to your pe matrix
        self.register_buffer("pe", pe)

        self.embedding_channel = nn.Conv1d(
            in_channels=self.d_model * self.patch_num1,
            out_channels=self.d_model,
            kernel_size=1,
        )

        self.embedding_patch_1 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len1,
            stride=self.stride1,
        )
        self.embedding_patch_2 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len2,
            stride=self.stride2,
        )
        self.embedding_patch_3 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len3,
            stride=self.stride3,
        )
        self.embedding_patch_4 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len4,
            stride=self.stride4,
        )

        self.out_linear_1 = torch.nn.Linear(self.d_model, self.pred_len // 8)
        self.out_linear_2 = torch.nn.Linear(
            self.d_model + self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_3 = torch.nn.Linear(
            self.d_model + 2 * self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_4 = torch.nn.Linear(
            self.d_model + 3 * self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_5 = torch.nn.Linear(
            self.d_model + self.pred_len // 2, self.pred_len // 8
        )
        self.out_linear_6 = torch.nn.Linear(
            self.d_model + 5 * self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_7 = torch.nn.Linear(
            self.d_model + 6 * self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_8 = torch.nn.Linear(
            self.d_model + 7 * self.pred_len // 8,
            self.pred_len - 7 * (self.pred_len // 8),
        )

        self.remap = torch.nn.Linear(self.d_model, self.seq_len)
        self.final_linear = nn.Linear(n_fea, 1)

    def forecast(self, x_enc):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Multi-scale embedding
        x_i = x_enc.permute(0, 2, 1)

        x_i_p1 = x_i
        x_i_p2 = self.padding_patch_layer2(x_i)
        x_i_p3 = self.padding_patch_layer3(x_i)
        x_i_p4 = self.padding_patch_layer4(x_i)
        encoding_patch1 = self.embedding_patch_1(
            rearrange(x_i_p1, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoding_patch2 = self.embedding_patch_2(
            rearrange(x_i_p2, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoding_patch3 = self.embedding_patch_3(
            rearrange(x_i_p3, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoding_patch4 = self.embedding_patch_4(
            rearrange(x_i_p4, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)

        encoding_patch = (
                torch.cat(
                    (encoding_patch1, encoding_patch2, encoding_patch3, encoding_patch4),
                    dim=-1,
                )
                + self.pe
        )
        # Temporal encoding
        for i in range(self.N):
            encoding_patch = self.encoder_list[i](encoding_patch)[0]

        # Channel-wise encoding
        x_patch_c = rearrange(
            encoding_patch, "(b c) p d -> b c (p d)", b=x_enc.shape[0], c=self.d_channel
        )
        x_ch = self.embedding_channel(x_patch_c.permute(0, 2, 1)).transpose(
            1, 2
        )  # [b c d]

        encoding_1_ch = self.encoder_list_ch[0](x_ch)[0]

        # Semi Auto-regressive
        forecast_ch1 = self.out_linear_1(encoding_1_ch)
        forecast_ch2 = self.out_linear_2(
            torch.cat((encoding_1_ch, forecast_ch1), dim=-1)
        )
        forecast_ch3 = self.out_linear_3(
            torch.cat((encoding_1_ch, forecast_ch1, forecast_ch2), dim=-1)
        )
        forecast_ch4 = self.out_linear_4(
            torch.cat((encoding_1_ch, forecast_ch1, forecast_ch2, forecast_ch3), dim=-1)
        )
        forecast_ch5 = self.out_linear_5(
            torch.cat(
                (encoding_1_ch, forecast_ch1, forecast_ch2, forecast_ch3, forecast_ch4),
                dim=-1,
            )
        )
        forecast_ch6 = self.out_linear_6(
            torch.cat(
                (
                    encoding_1_ch,
                    forecast_ch1,
                    forecast_ch2,
                    forecast_ch3,
                    forecast_ch4,
                    forecast_ch5,
                ),
                dim=-1,
            )
        )
        forecast_ch7 = self.out_linear_7(
            torch.cat(
                (
                    encoding_1_ch,
                    forecast_ch1,
                    forecast_ch2,
                    forecast_ch3,
                    forecast_ch4,
                    forecast_ch5,
                    forecast_ch6,
                ),
                dim=-1,
            )
        )
        forecast_ch8 = self.out_linear_8(
            torch.cat(
                (
                    encoding_1_ch,
                    forecast_ch1,
                    forecast_ch2,
                    forecast_ch3,
                    forecast_ch4,
                    forecast_ch5,
                    forecast_ch6,
                    forecast_ch7,
                ),
                dim=-1,
            )
        )

        final_forecast = torch.cat(
            (
                forecast_ch1,
                forecast_ch2,
                forecast_ch3,
                forecast_ch4,
                forecast_ch5,
                forecast_ch6,
                forecast_ch7,
                forecast_ch8,
            ),
            dim=-1,
        ).permute(0, 2, 1)

        # De-Normalization
        dec_out = final_forecast * (
            stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, feat, extra_feat=None):
        x = feat.unsqueeze(-1)
        if extra_feat is not None:
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        B_ori, node, seq_len, channel = x.shape
        x_enc = x.view(B_ori * node, seq_len, channel)
        outputs = []
        current_chunk_size = self.chunk_size
        start_idx = 0
        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx: start_idx + current_chunk_size]
                dec_out = self.forecast(x_chunk)
                dec_out = self.final_linear(dec_out)
                dec_out = dec_out.squeeze(-1)
                outputs.append(dec_out)
                start_idx += current_chunk_size
            except RuntimeError as e:
                if "CUDA" in str(e):
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e
        out = torch.cat(outputs, dim=0)
        out = out.view(B_ori, node)
        return out


class ConvTimeNet(nn.Module):
    def __init__(self, c_in: int, c_out: int, seq_len: int, n_layers: int = 3, d_model: int = 128,
                 d_ff: int = 256, dropout=0.1, act: str = "relu", pooling_tp='max', fc_dropout: float = 0.,
                 enable_res_param=False, dw_ks=[7, 13, 19], norm='batch', use_embed=True, re_param=False,
                 device: str = 'cuda:0'):

        super(ConvTimeNet, self).__init__()
        assert n_layers == len(dw_ks), "dw_ks should match the n_layers!"

        self.c_out, self.seq_len = c_out, seq_len

        # Input Embedding
        self.use_embed = use_embed
        self.W_P = nn.Linear(c_in, d_model)

        # Positional encoding
        # W_pos = torch.empty((seq_len, d_model), device=device)
        # nn.init.uniform_(W_pos, -0.02, 0.02)
        # self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = ConvTimeNet_ConvEncoder(d_model, d_ff, kernel_size=dw_ks, dropout=dropout, activation=act, \
                                               n_layers=n_layers, enable_res_param=enable_res_param, norm=norm,
                                               re_param=re_param,
                                               device=device)

        self.flatten = nn.Flatten()

        # Head
        self.head_nf = seq_len * d_model if pooling_tp == 'cat' else d_model
        self.head = self.create_head(self.head_nf, c_out, act=act, pooling_tp=pooling_tp, fc_dropout=fc_dropout)

    def create_head(self, nf, c_out, act="gelu", pooling_tp='max', fc_dropout=0., **kwargs):
        layers = []
        if pooling_tp == 'cat':
            layers = [ConvTimeNet_get_activation_fn(act), self.flatten]
            if fc_dropout: layers += [nn.Dropout(fc_dropout)]
        elif pooling_tp == 'mean':
            layers = [nn.AdaptiveAvgPool1d(1), self.flatten]
        elif pooling_tp == 'max':
            layers = [nn.AdaptiveMaxPool1d(1), self.flatten]

        layers += [nn.Linear(nf, c_out)]

        # could just be used in classifying task
        return nn.Sequential(*layers)

    def forward(self, feat, extra_feat=None) -> torch.Tensor:
        x = feat.unsqueeze(-1)
        if extra_feat is not None:
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        B_ori, node, seq_len, channel = x.shape
        x_enc = x.view(B_ori * node, seq_len, channel)
        outputs = []
        current_chunk_size = self.chunk_size
        start_idx = 0
        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx: start_idx + current_chunk_size]
                x_chunk = x_chunk.permute(0, 2, 1)
                if self.use_embed:
                    u = self.W_P(x_chunk.transpose(2, 1))
                z = self.encoder(u.transpose(2, 1).contiguous())
                out = self.head(z)
                outputs.append(out)
                start_idx += current_chunk_size
            except RuntimeError as e:
                if "CUDA" in str(e):
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e
        out = torch.cat(outputs, dim=0)
        out = out.view(B_ori, node)
        return out
