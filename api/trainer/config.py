# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/11 2:56
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/11 2:56
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from api.utils import calculate_regression_metrics, convert_numpy


class PredictionTrainer(object):
    def __init__(
            self,
            dataset,
            model,
            seq_l,
            pre_len,
            is_train,
            save_path,
    ):
        self.ev_dataset = dataset
        self.ev_model = model
        self.save_path = save_path
        self.is_train = is_train
        if self.ev_model.model_name == 'lo' or self.ev_model.model_name == 'ar' or self.ev_model.model_name == 'arima':
            self.optim = None
            self.loss_func = None
            self.is_train = False
            self.stat_model = True
            self.train_valid_feat = np.vstack(
                (dataset.train_feat, dataset.valid_feat, dataset.test_feat[:seq_l + pre_len, :]))
            self.test_loader = [self.train_valid_feat, dataset.test_feat[pre_len + seq_l:, :]]
        else:
            self.optim = torch.optim.Adam(self.ev_model.model.parameters(), weight_decay=0.00001)
            self.stat_model = False
            self.loss_func = torch.nn.MSELoss()

    def training(
            self,
            epoch,
    ):
        valid_loss = 1000
        self.ev_model.model.train()
        for _ in tqdm(range(epoch), desc='Training'):
            for j, data in enumerate(self.ev_dataset.train_loader):
                feat, label, extra_feat = data
                if self.ev_dataset.extra_feat is None:
                    extra_feat=None
                self.optim.zero_grad()
                predict = self.ev_model.model(feat, extra_feat)
                if predict.shape != label.shape:
                    loss = self.loss_func(predict.unsqueeze(-1), label)
                else:
                    loss = self.loss_func(predict, label)
                loss.backward()
                self.optim.step()

            # validation
            for j, data in enumerate(self.ev_dataset.valid_loader):
                feat, label, extra_feat = data
                if self.ev_dataset.extra_feat is None:
                    extra_feat=None

                predict = self.ev_model.model(feat, extra_feat)
                if predict.shape != label.shape:
                    loss = self.loss_func(predict.unsqueeze(-1), label)
                else:
                    loss = self.loss_func(predict, label)
                if loss.item() < valid_loss:
                    valid_loss = loss.item()
                    path = os.path.join(self.save_path,'train.pth')
                    torch.save(self.ev_model.model.state_dict(), path)

    def test(
            self,
            model_path=None,
    ):
        predict_list = []
        label_list = []

        if not self.stat_model:
            if model_path is not None:
                self.ev_model.load_model(model_path=model_path)
            self.ev_model.model.eval()
            for j, data in enumerate(self.ev_dataset.test_loader):
                feat, label, extra_feat = data
                if self.ev_dataset.extra_feat is None:
                    extra_feat=None

                with torch.no_grad():
                    predict = self.ev_model.model(feat, extra_feat)
                    if predict.shape != label.shape:
                        predict = predict.unsqueeze(-1)
                    predict = predict.cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                predict_list.append(predict)
                label_list.append(label)

        else:
            train_valid_occ, test_occ = self.test_loader
            predict = self.ev_model.model.predict(train_valid_occ, test_occ)
            label = test_occ
            predict_list.append(predict)
            label_list.append(label)

        predict_array = np.concatenate(predict_list, axis=0)
        label_array = np.concatenate(label_list, axis=0)


        if self.ev_dataset.scaler is not None:
            predict_array = self.ev_dataset.scaler.inverse_transform(predict_array)
            label_array = self.ev_dataset.scaler.inverse_transform(label_array)

        np.save(os.path.join(self.save_path,'predict.npy'),predict_array)
        np.save(os.path.join(self.save_path,'label.npy'),label_array)
        result_metrics=calculate_regression_metrics(y_true=label_array,y_pred=predict_array)
        result_metrics_converted = convert_numpy(result_metrics)
        with open(os.path.join(self.save_path,'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(result_metrics_converted, f, ensure_ascii=False, indent=4)

