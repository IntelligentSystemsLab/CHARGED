# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/11 2:03
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/11 2:03
import torch

from api.model.methods import Lstm, Lo, Ar, Arima, Fcnn, SegRNN, FreTS, ModernTCN, MultiPatchFormer


class PredictionModel(object):
    def __init__(
        self,
        num_node,
        n_fea,
        model_name,
        seq_l,
        pre_len,
        device,
    ):
        self.model_name=model_name
        if model_name == 'lstm':
            self.model = Lstm(seq_l, n_fea, node=num_node)
        elif model_name == 'lo':
            self.model = Lo(pre_len)
        elif model_name == 'ar':
            self.model = Ar(pred_len=pre_len, lags=seq_l)
        elif model_name == 'arima':
            self.model = Arima(pred_len=pre_len)
        elif model_name == 'fcnn':
            self.model = Fcnn(n_fea, node=num_node, seq=seq_l)
        elif model_name == 'segrnn':
            self.model = SegRNN(seq_len=seq_l,pred_len=pre_len,n_fea=n_fea)
        elif model_name == 'frets':
            self.model = FreTS(seq_len=seq_l,pred_len=pre_len,n_fea=n_fea)
        elif model_name == 'moderntcn':
            self.model = ModernTCN(seq_len=seq_l,n_fea=n_fea,pred_len=pre_len)
        elif model_name == 'multipatchformer':
            self.model = MultiPatchFormer(seq_len=seq_l,n_fea=n_fea,pred_len=pre_len)
        self.model.chunk_size=512

    def load_model(self,model_path):
        state_dict = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(state_dict)

    def update_chunksize(self,chunk_size):
        self.model.chunk_size=chunk_size
