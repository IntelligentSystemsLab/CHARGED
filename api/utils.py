# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 16:20
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 16:20


"""
Utility module providing logging, reproducibility, dataset creation, and evaluation metrics for EV demand prediction.

Functions:
    Logger: Redirects stdout to both terminal and log file with timestamps.
    random_seed: Sets random seeds for Python, NumPy, and Torch for reproducibility.
    get_n_feature: Computes input feature count including auxiliary features.
    CreateDataset: PyTorch Dataset for sequence-to-one prediction, handling auxiliary features.
    create_rnn_data: Generates sliding-window samples and labels from time-series data.
    calculate_regression_metrics: Computes regression metrics (MAE, RMSE, MAPE, RAE, MedAE, R², EVS).
    convert_numpy: Recursively converts NumPy types in structures to native Python types.
    get_data_paths: Constructs per-city data directory paths.
"""
import datetime
import os
import random
import sys
from typing import Any, Optional, Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    mean_absolute_percentage_error,
)


class Logger(object):
    """
    Logs stdout messages to both console and file with timestamps.

    Args:
        filename (str): Path to the log file.
    """

    def __init__(self, filename: str = "log.txt") -> None:
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        """
        Write message to console and log file, prepending a timestamp if non-empty.

        Args:
            message (str): Message to log.
        """
        if message.strip():
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"{timestamp} - {message}"
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        """
        Flush both console and file streams.
        """
        self.terminal.flush()
        self.log.flush()


def random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across torch, NumPy, and Python.

    Args:
        seed (int): Seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_n_feature(extra_feat: Optional[np.ndarray]) -> int:
    """
    Determine the number of input feature channels including auxiliary.

    Args:
        extra_feat (np.ndarray or None): Auxiliary features [T, N, C] or None.

    Returns:
        int: Total feature channels (1 base + C auxiliary, or 1 if no auxiliary).
    """
    return 1 if extra_feat is None else extra_feat.shape[-1] + 1


class CreateDataset(Dataset):
    """
    PyTorch Dataset for sliding-window RNN data.

    Converts features and optional auxiliary data into sequence-label pairs and
    moves tensors to the specified device.

    Args:
        seq_l (int): Length of input sequence.
        pre_len (int): Prediction horizon.
        feat (np.ndarray): Feature series [T, N].
        extra_feat (np.ndarray or None): Auxiliary series [T, N, C].
        device (torch.device): Target device for tensors.
    """

    def __init__(
            self,
            seq_l: int,
            pre_len: int,
            feat: np.ndarray,
            extra_feat: Optional[np.ndarray],
            device: torch.device,
    ) -> None:
        x, y = create_rnn_data(feat, seq_l, pre_len)
        self.feat = torch.Tensor(x)
        self.label = torch.Tensor(y)
        self.extra_feat = None
        if extra_feat is not None:
            x2, _ = create_rnn_data(extra_feat, seq_l, pre_len)
            self.extra_feat = torch.Tensor(x2)
        self.device = device

    def __len__(self) -> int:
        return len(self.feat)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample and label, transposed to [N, seq_l] format.

        Returns:
            feat (torch.Tensor): [N, seq_l] input.
            label (torch.Tensor): Label tensor.
            extra_feat (torch.Tensor): [N, seq_l] auxiliary or empty tensor.
        """
        feat = self.feat[idx].transpose(0, 1).to(self.device)
        label = self.label[idx].to(self.device)
        if self.extra_feat is not None:
            ef = self.extra_feat[idx].transpose(0, 1).to(self.device)
            return feat, label, ef
        dummy = torch.empty(0, device=self.device)
        return feat, label, dummy


def create_rnn_data(
        dataset: np.ndarray,
        lookback: int,
        predict_time: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences and corresponding labels.

    Args:
        dataset (np.ndarray): Time-series data [T, ...].
        lookback (int): Number of past steps.
        predict_time (int): Steps ahead to predict.

    Returns:
        x (np.ndarray): Array of shape [num_samples, lookback, ...].
        y (np.ndarray): Array of shape [num_samples, ...] with target at lookback+predict_time-1.
    """
    x, y = [], []
    L = len(dataset)
    for i in range(L - lookback - predict_time):
        x.append(dataset[i: i + lookback])
        y.append(dataset[i + lookback + predict_time - 1])
    return np.array(x), np.array(y)


def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        optimized: bool = True,
) -> Dict[str, float]:
    """
    Compute common regression metrics.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        optimized (bool): If True, applies small-epsilon adjustments.

    Returns:
        dict: Metrics {MAE, RMSE, MAPE, RAE, MedAE, R², EVS}.
    """
    eps = 1e-6
    if optimized:
        yt = y_true.copy()
        yp = y_pred.copy()
        yt[yt <= eps] = np.abs(yt[yt <= eps]) + eps
        yp[yt <= eps] = np.abs(yp[yt <= eps]) + eps
        mape = mean_absolute_percentage_error(yt, yp)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + eps)
        evs = 1 - np.var(y_true - y_pred) / (np.var(y_true) + eps)
    else:
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if np.sum(np.abs(y_true - np.mean(y_true))) == 0:
        rae = np.sum(np.abs(y_true - y_pred)) / eps
    else:
        rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
    medae = median_absolute_error(y_true, y_pred)
    return {
        'MAE'  : mae,
        'RMSE' : rmse,
        'MAPE' : mape,
        'RAE'  : rae,
        'MedAE': medae,
        'R²'   : r2,
        'EVS'  : evs,
    }


def convert_numpy(obj: Any) -> Any:
    """
    Recursively convert NumPy scalar types in a structure to Python native types.

    Args:
        obj: np.generic, dict, list, or other.

    Returns:
        Object with np types converted to native.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


def get_data_paths(
        ori_path: str,
        cities: str,
        suffix: str = '_remove_zero',
) -> Dict[str, str]:
    """
    Generate data directory paths for multiple cities.

    Args:
        ori_path (str): Base directory path.
        cities (str): '+'-separated city codes.
        suffix (str): Directory suffix per city (default '_remove_zero').

    Returns:
        dict: Mapping city code to full data path.
    """
    paths = {}
    for city in cities.split('+'):
        paths[city] = os.path.join(ori_path, f"{city}{suffix}/")
    return paths
