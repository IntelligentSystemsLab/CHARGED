# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 17:16
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 17:16

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

from api.utils import CreateDataset


class EVDataset(object):
    def __init__(
            self,
            feature,
            auxiliary,
            data_path,
    ):
        super(EVDataset, self).__init__()
        self.feature = feature
        self.auxiliary = auxiliary
        self.data_path = data_path
        if self.feature == 'volume':
            self.feat = pd.read_csv(f'{self.data_path}volume.csv', header=0, index_col=0)
        elif self.feature == 'duration':
            self.feat = pd.read_csv(f'{self.data_path}duration.csv', header=0, index_col=0)

        self.e_price = pd.read_csv(f'{self.data_path}e_price.csv', index_col=0, header=0).values
        self.s_price = pd.read_csv(f'{self.data_path}s_price.csv', index_col=0, header=0).values

        self.time = pd.to_datetime(self.feat.index)

        price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.e_price = price_scaler.fit_transform(self.e_price)
        self.s_price = price_scaler.fit_transform(self.s_price)

        self.weather = pd.read_csv(f'{self.data_path}weather.csv', header=0, index_col='time')
        self.weather = self.weather[['temp', 'precip', 'visibility']]

        self.extra_feat = None
        if self.auxiliary != 'None':
            self.extra_feat = np.zeros([self.feat.shape[0], self.feat.shape[1], 1])
            if self.auxiliary == 'all':
                self.extra_feat = np.concatenate([self.extra_feat, self.e_price[:, :, np.newaxis]], axis=2)
                self.extra_feat = np.concatenate([self.extra_feat, self.s_price[:, :, np.newaxis]], axis=2)
                self.extra_feat = np.concatenate([self.extra_feat,
                                                  np.repeat(self.weather.values[:, np.newaxis, :], self.feat.shape[1],
                                                            axis=1)],
                                                 axis=2)
            else:
                add_feat_list = self.auxiliary.split('+')
                for add_feat in add_feat_list:
                    if add_feat == 'e_price':
                        self.extra_feat = np.concatenate([self.extra_feat, self.e_price[:, :, np.newaxis]], axis=2)
                    elif add_feat == 's_price':
                        self.extra_feat = np.concatenate([self.extra_feat, self.s_price[:, :, np.newaxis]], axis=2)
                    else:
                        self.extra_feat = np.concatenate([self.extra_feat,
                                                          np.repeat(
                                                              self.weather[add_feat].values[:, np.newaxis, np.newaxis],
                                                              self.feat.shape[1], axis=1)], axis=2)
            self.extra_feat = self.extra_feat[:, :, 1:]
        self.feat = np.array(self.feat)

    def split_cross_validation(
            self,
            fold,
            total_fold,
            train_ratio,
            valid_ratio,
            pred_type,
    ):
        assert len(self.time) == len(self.feat)
        month_list = sorted(np.unique(self.time.month))
        assert total_fold == len(month_list)
        fold_time = self.time.month.isin(month_list[0:fold]).sum()
        train_end = int(fold_time * train_ratio)
        valid_start = train_end
        valid_end = int(valid_start + fold_time * valid_ratio)
        test_start = valid_end
        test_end = int(fold_time)
        train_feat = self.feat[:train_end]
        valid_feat = self.feat[valid_start:valid_end]
        test_feat = self.feat[test_start:test_end]

        self.scaler = None

        if pred_type == 'region':
            self.scaler = StandardScaler()
            self.train_feat = self.scaler.fit_transform(train_feat)
            self.valid_feat = self.scaler.transform(valid_feat)
            self.test_feat = self.scaler.transform(test_feat)
        else:
            node_idx = int(pred_type)
            self.scaler = StandardScaler()
            self.train_feat = self.scaler.fit_transform(train_feat[:, node_idx].reshape(-1, 1))
            self.valid_feat = self.scaler.transform(valid_feat[:, node_idx].reshape(-1, 1))
            self.test_feat = self.scaler.transform(test_feat[:, node_idx].reshape(-1, 1))

        self.train_extra_feat, self.valid_extra_feat, self.test_extra_feat = None, None, None
        if self.extra_feat is not None:
            self.train_extra_feat = self.extra_feat[:train_end]
            self.valid_extra_feat = self.extra_feat[valid_start:valid_end]
            self.test_extra_feat = self.extra_feat[test_start:test_end]

        assert len(train_feat) > 0, "The training set cannot be empty!"

    def create_loaders(
            self,
            seq_l,
            pre_len,
            batch_size,
            device,
    ):

        train_dataset = CreateDataset(seq_l,pre_len,self.train_feat, self.train_extra_feat, device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        valid_dataset = CreateDataset(seq_l,pre_len,self.valid_feat, self.valid_extra_feat, device)
        self.valid_loader = DataLoader(valid_dataset, batch_size=len(self.valid_feat), shuffle=False)

        test_dataset = CreateDataset(seq_l,pre_len,self.test_feat, self.test_extra_feat, device)
        self.test_loader = DataLoader(test_dataset, batch_size=len(self.test_feat), shuffle=False)
