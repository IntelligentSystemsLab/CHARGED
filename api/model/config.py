# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/11 2:03
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/11 2:03

import torch
from typing import Any

from api.model.modules import Lstm, Lo, Ar, Arima, Fcnn, SegRNN, FreTS, ModernTCN, MultiPatchFormer, ConvTimeNet

"""
Module defining the PredictionModel factory for instantiating various time-series prediction architectures.

Supported models include LSTM, LO, AR, ARIMA, FCNN, SegRNN, FreTS, ModernTCN, MultiPatchFormer, and ConvTimeNet.
Each model is initialized with sequence length, feature dimensions, nodes, and prediction horizon as appropriate.
"""


class PredictionModel(object):
    """
    Factory and loader for prediction models.

    This class selects and initializes a specific model architecture based on the provided name,
    and provides utility methods for loading pretrained weights and updating model settings.

    Attributes:
        model_name (str): Identifier of the model architecture.
        model (torch.nn.Module): Instantiated model ready for training or inference.
    """

    def __init__(
            self,
            num_node: int,
            n_fea: int,
            model_name: str,
            seq_l: int,
            pre_len: int,
    ) -> None:
        """
        Initialize a PredictionModel instance by selecting the appropriate architecture.

        Args:
            num_node (int): Number of nodes or output dimensions (for graph-based models).
            n_fea (int): Number of input feature channels.
            model_name (str): Name of the model to instantiate ('lstm', 'lo', 'ar', etc.).
            seq_l (int): Input sequence length.
            pre_len (int): Prediction horizon length.

        Raises:
            ValueError: If the specified model_name is unsupported.
        """
        self.model_name = model_name

        if model_name == 'lstm':
            self.model = Lstm(seq_l=seq_l, n_feature=n_fea, node=num_node)
        elif model_name == 'lo':
            self.model = Lo(pred_len=pre_len)
        elif model_name == 'ar':
            self.model = Ar(pred_len=pre_len, lags=seq_l)
        elif model_name == 'arima':
            self.model = Arima(pred_len=pre_len)
        elif model_name == 'fcnn':
            self.model = Fcnn(n_fea, node=num_node, seq=seq_l)
        elif model_name == 'segrnn':
            self.model = SegRNN(seq_len=seq_l, pred_len=pre_len, n_fea=n_fea)
        elif model_name == 'frets':
            self.model = FreTS(seq_len=seq_l, pred_len=pre_len, n_fea=n_fea)
        elif model_name == 'moderntcn':
            self.model = ModernTCN(seq_len=seq_l, n_fea=n_fea, pred_len=pre_len)
        elif model_name == 'multipatchformer':
            self.model = MultiPatchFormer(seq_len=seq_l, n_fea=n_fea, pred_len=pre_len)
        elif model_name == 'convtimenet':
            self.model = ConvTimeNet(seq_len=seq_l, c_in=n_fea, c_out=pre_len)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # Default chunk size for processing large sequences in batches
        self.model.chunk_size = 512

    def load_model(self, model_path: str) -> None:
        """
        Load pretrained model weights from a checkpoint file.

        Args:
            model_path (str): Filesystem path to the saved state dictionary.
        """
        state_dict = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(state_dict)

    def update_chunksize(self, chunk_size: int) -> None:
        """
        Update the model's internal chunk size parameter.

        This can control memory usage and throughput when processing long sequences.

        Args:
            chunk_size (int): New chunk size for sequence processing.
        """
        self.model.chunk_size = chunk_size
