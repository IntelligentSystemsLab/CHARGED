# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/12 15:41
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/12 15:41

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

"""
Module defining attention layers and modern TCN/transformer-inspired blocks
for time-series forecasting models.

Classes:
    AttentionLayer: Multi-head attention wrapper with projections.
    FullAttention: Standard scaled dot-product attention with optional mask.
    TriangularCausalMask: Generates causal mask for autoregressive attention.
    ModernTCN_moving_avg: Moving average block for trend extraction.
    ModernTCN_series_decomp: Time-series decomposition into trend and residual.
    ModernTCN_Flatten_Head: Forecasting head flattening TCN output.
    ModernTCN_RevIN: Reversible Instance Normalization.
    ModernTCN_LayerNorm: Layer normalization for TCN outputs.
    ModernTCN_ReparamLargeKernelConv: Re-parameterizable large-kernel conv block.
    ModernTCN_Block: Single TCN block with depthwise and pointwise conv.
    ModernTCN_Stage: Stack of TCN blocks.
    MultiPatchFormer_FeedForward: FFN for transformer encoder.
    MultiPatchFormer_Encoder: Transformer-like encoder module.
    ConvTimeNet_ConvEncoder: Stack of convolutional encoder layers.
    ConvTimeNet_ConvEncoderLayer: Single conv encoder layer with optional re-param.
    ConvTimeNet_SublayerConnection: Residual connection with optional parameter.

Utility Functions:
    ModernTCN_get_conv1d, ModernTCN_get_bn, ModernTCN_conv_bn,
    ModernTCN_fuse_bn, ConvTimeNet_get_activation_fn
"""


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ModernTCN_moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(ModernTCN_moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class ModernTCN_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(ModernTCN_series_decomp, self).__init__()
        self.moving_avg = ModernTCN_moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# forecast task head
class ModernTCN_Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super(ModernTCN_Flatten_Head, self).__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class ModernTCN_RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(ModernTCN_RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class ModernTCN_LayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(ModernTCN_LayerNorm, self).__init__()
        self.norm = nn.Layernorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(
            x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x


def ModernTCN_get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def ModernTCN_get_bn(channels):
    return nn.BatchNorm1d(channels)


def ModernTCN_conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv',
                      ModernTCN_get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', ModernTCN_get_bn(out_channels))
    return result


def ModernTCN_fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ModernTCN_ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ModernTCN_ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = ModernTCN_conv_bn(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = ModernTCN_conv_bn(in_channels=in_channels, out_channels=out_channels,
                                                    kernel_size=small_kernel,
                                                    stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,
                                                    bias=False)

    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self, x, pad_length_left, pad_length_right, pad_values=0):

        D_out, D_in, ks = x.shape
        if pad_values == 0:
            pad_left = torch.zeros(D_out, D_in, pad_length_left)
            pad_right = torch.zeros(D_out, D_in, pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left, x], dims=-1)
        x = torch.cat([x, pad_right], dims=-1)
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = ModernTCN_fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = ModernTCN_fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class ModernTCN_Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):
        super(ModernTCN_Block, self).__init__()
        self.dw = ModernTCN_ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                                   kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                                   small_kernel=small_size, small_kernel_merged=small_kernel_merged,
                                                   nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)

        # convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff // dmodel

    def forward(self, x):
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = input + x
        return x


class ModernTCN_Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):

        super(ModernTCN_Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = ModernTCN_Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars,
                                  small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


class MultiPatchFormer_FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 512):
        super(MultiPatchFormer_FeedForward, self).__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)

        return x


class MultiPatchFormer_Encoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            mha: AttentionLayer,
            d_hidden: int,
            dropout: float = 0,
            channel_wise=False,
    ):
        super(MultiPatchFormer_Encoder, self).__init__()

        self.channel_wise = channel_wise
        if self.channel_wise:
            self.conv = torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="reflect",
            )
        self.MHA = mha
        self.feedforward = MultiPatchFormer_FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        q = residual
        if self.channel_wise:
            x_r = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
            k = x_r
            v = x_r
        else:
            k = residual
            v = residual
        x, score = self.MHA(q, k, v, attn_mask=None)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(residual)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score


class ConvTimeNet_ConvEncoder(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size=[19, 19, 29, 29, 37, 37], dropout=0.1, activation='gelu',
                 n_layers=3, enable_res_param=False, norm='batch', re_param=False, device='cuda:0'):
        super(ConvTimeNet_ConvEncoder, self).__init__()
        self.layers = nn.ModuleList([ConvTimeNet_ConvEncoderLayer(kernel_size[i], d_model, d_ff=d_ff, dropout=dropout,
                                                                  activation=activation,
                                                                  enable_res_param=enable_res_param,
                                                                  norm=norm,
                                                                  re_param=re_param, device=device) \
                                     for i in range(n_layers)])

    def forward(self, src):
        output = src
        for mod in self.layers: output = mod(output)
        return output


class ConvTimeNet_ConvEncoderLayer(nn.Module):
    def __init__(self, kernel_size, d_model, d_ff=256, dropout=0.1, activation="relu",
                 enable_res_param=True, norm='batch', small_ks=3, re_param=True, device='cuda:0'):
        super(ConvTimeNet_ConvEncoderLayer, self).__init__()

        self.norm_tp = norm
        self.re_param = re_param

        # DeepWise Conv. Add & Norm
        if self.re_param:
            self.large_ks = kernel_size
            self.small_ks = small_ks
            self.DW_conv_large = nn.Conv1d(d_model, d_model, self.large_ks, stride=1, padding=self.large_ks // 2,
                                           groups=d_model)
            self.DW_conv_small = nn.Conv1d(d_model, d_model, self.small_ks, stride=1, padding=self.small_ks // 2,
                                           groups=d_model)
            self.DW_infer = nn.Conv1d(d_model, d_model, self.large_ks, stride=1, padding=self.large_ks // 2,
                                      groups=d_model)
        else:
            self.DW_conv = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding=kernel_size // 2, groups=d_model)

        self.dw_act = ConvTimeNet_get_activation_fn(activation)

        self.sublayerconnect1 = ConvTimeNet_SublayerConnection(enable_res_param, dropout)
        self.dw_norm = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Conv1d(d_model, d_ff, 1, 1),
                                ConvTimeNet_get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Conv1d(d_ff, d_model, 1, 1))

        # Add & Norm
        self.sublayerconnect2 = ConvTimeNet_SublayerConnection(enable_res_param, dropout)
        self.norm_ffn = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)

    def _get_merge_param(self):
        left_pad = (self.large_ks - self.small_ks) // 2
        right_pad = (self.large_ks - self.small_ks) - left_pad
        module_output = copy.deepcopy(self.DW_conv_large)
        module_output.weight += F.pad(self.DW_conv_small.weight, (left_pad, right_pad), value=0)
        module_output.bias += self.DW_conv_small.bias
        self.DW_infer = module_output

    def forward(self, src):  # [B, C, L]

        ## Deep-wise Conv Layer
        if not self.re_param:
            src = self.DW_conv(src)
        else:
            if self.training:  # training phase
                large_out, small_out = self.DW_conv_large(src), self.DW_conv_small(src)
                src = self.sublayerconnect1(src, self.dw_act(large_out + small_out))
            else:  # testing phase
                self._get_merge_param()
                merge_out = self.DW_infer(src)
                src = self.sublayerconnect1(src, self.dw_act(merge_out))

        src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src
        src = self.dw_norm(src)
        src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src

        ## Position-wise Conv Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm

        src2 = self.sublayerconnect2(src, src2)  # Add: residual connection with residual dropout

        # Norm: batchnorm or layernorm
        src2 = src2.permute(0, 2, 1) if self.norm_tp != 'batch' else src2
        src2 = self.norm_ffn(src2)
        src2 = src2.permute(0, 2, 1) if self.norm_tp != 'batch' else src2

        return src


def ConvTimeNet_get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        return activation()


class ConvTimeNet_SublayerConnection(nn.Module):

    def __init__(self, enable_res_parameter, dropout=0.1):
        super(ConvTimeNet_SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, out_x):
        if not self.enable:
            return x + self.dropout(out_x)
        else:
            # print(self.a)
            # print(torch.mean(torch.abs(x) / torch.abs(out_x)))
            return x + self.dropout(self.a * out_x)
