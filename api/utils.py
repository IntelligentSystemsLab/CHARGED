# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 16:20
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 16:20
import datetime
import random
import sys

import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, \
    explained_variance_score, mean_absolute_percentage_error

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        # 如果 message 为空行则直接写入（避免重复添加时间戳）
        if message.strip():
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"{timestamp} - {message}"
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_n_feature(extra_feat):
    if extra_feat is None:
        return 1
    else:
        return extra_feat.shape[-1]+1


class CreateDataset(Dataset):
    def __init__(
            self,
            seq_l,
            pre_len,
            feat,
            extra_feat,
            device
    ):
        lb = seq_l
        pt = pre_len
        feat, label = create_rnn_data(feat, lb, pt)
        self.feat = torch.Tensor(feat)
        self.label = torch.Tensor(label)

        self.extra_feat = None
        if extra_feat is not None:
            extra_feat, _ = create_rnn_data(extra_feat, lb, pt)
            self.extra_feat = torch.Tensor(extra_feat)
        self.device = device

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, idx):  # feat: batch, seq, node
        output_feat = torch.transpose(self.feat[idx, :, :], 0, 1).to(self.device)
        output_label = self.label[idx, :].to(self.device)
        if self.extra_feat is not None:
            output_extra_feat = torch.transpose(self.extra_feat[idx, :, :], 0, 1).to(self.device)
            return output_feat, output_label, output_extra_feat
        else:
            dummy_extra = torch.empty(0, device=self.device)
            return output_feat, output_label,dummy_extra


def create_rnn_data(dataset, lookback, predict_time):
    x = []
    y = []
    for i in range(len(dataset) - lookback - predict_time):
        x.append(dataset[i:i + lookback])
        y.append(dataset[i + lookback + predict_time - 1])
    return np.array(x), np.array(y)

def calculate_regression_metrics(y_true, y_pred):
    eps = 2e-2
    MAPE_y_true = y_true.copy()
    MAPE_y_pred = y_pred.copy()
    MAPE_y_true[np.where(MAPE_y_true <= eps)] = np.abs(MAPE_y_true[np.where(MAPE_y_true <= eps)]) + eps
    MAPE_y_pred[np.where(MAPE_y_true <= eps)] = np.abs(MAPE_y_pred[np.where(MAPE_y_true <= eps)]) + eps

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(MAPE_y_true, MAPE_y_pred)
    if np.sum(np.abs(y_true - np.mean(y_true)))==0:
        rae = np.sum(np.abs(y_true - y_pred)) / eps
    else:
        rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'RAE': rae,
        'MedAE': medae,
        'R²': r2,
        'EVS': evs
    }

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj

def get_data_paths(ori_path,cities,suffix='_remove_zero'):
    city_list = cities.split('+')
    data_paths={}
    for city in city_list:
        data_paths[city]=f'{ori_path}{city}{suffix}/'
    return data_paths