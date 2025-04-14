# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/15 1:33
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/15 1:33

import json
import os

import numpy as np
import torch
from tqdm import tqdm

from api.utils import calculate_regression_metrics, convert_numpy


class ClientTrainer(object):
    def __init__(
            self,
            train_loader,
            test_loader,
            extra_feat_tag,
            model,
            save_path,
            scaler,
            device,
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.extra_feat_tag = extra_feat_tag
        self.ev_model = model
        self.save_path = save_path
        self.device = device
        self.scaler = scaler
        self.now_epoch = 0
        self.deploy_epoch = 0
        if self.ev_model.model_name == 'lo' or self.ev_model.model_name == 'ar' or self.ev_model.model_name == 'arima':
            raise ValueError("Not applicable.")
        else:
            self.optim = torch.optim.Adam(self.ev_model.model.parameters(), weight_decay=0.00001)
            self.loss_func = torch.nn.MSELoss()

    def training(
            self,
            epoch,
            save_model=False
    ):
        self.ev_model.model.train()
        self.ev_model.model.to(self.device)
        for _ in tqdm(range(epoch), desc='Training'):
            for j, data in enumerate(self.train_loader):
                torch.cuda.empty_cache()
                feat, label, extra_feat = data
                if not self.extra_feat_tag:
                    extra_feat=None
                self.optim.zero_grad()
                predict = self.ev_model.model(feat, extra_feat)
                if predict.shape != label.shape:
                    loss = self.loss_func(predict.unsqueeze(-1), label)
                else:
                    loss = self.loss_func(predict, label)
                loss.backward()
                self.optim.step()
        if save_model:
            path = os.path.join(self.save_path,f'train_{self.now_epoch}_{self.deploy_epoch}.pth')
            torch.save(self.ev_model.model.state_dict(), path)
        self.ev_model.model.to('cpu')
        torch.cuda.empty_cache()

    def test(
            self,
            model_path=None,
    ):
        predict_list = []
        label_list = []
        self.ev_model.model.to(self.device)

        if model_path is not None:
            self.ev_model.load_model(model_path=model_path)
        self.ev_model.model.eval()
        for j, data in enumerate(self.test_loader):
            torch.cuda.empty_cache()
            feat, label, extra_feat = data
            if not self.extra_feat_tag:
                extra_feat=None

            with torch.no_grad():
                predict = self.ev_model.model(feat, extra_feat)
                if predict.shape != label.shape:
                    predict = predict.unsqueeze(-1)
                predict = predict.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            predict_list.append(predict)
            label_list.append(label)


        predict_array = np.concatenate(predict_list, axis=0)
        label_array = np.concatenate(label_list, axis=0)


        if self.scaler is not None:
            predict_array = predict_array * self.scaler
            label_array = label_array* self.scaler

        np.save(os.path.join(self.save_path,f'predict_{self.now_epoch}_{self.deploy_epoch}.npy'),predict_array)
        np.save(os.path.join(self.save_path,f'label_{self.now_epoch}_{self.deploy_epoch}.npy'),label_array)
        result_metrics=calculate_regression_metrics(y_true=label_array,y_pred=predict_array)
        result_metrics_converted = convert_numpy(result_metrics)
        with open(os.path.join(self.save_path,f'metrics_{self.now_epoch}_{self.deploy_epoch}.json'), 'w', encoding='utf-8') as f:
            json.dump(result_metrics_converted, f, ensure_ascii=False, indent=4)

        self.ev_model.model.to('cpu')
        torch.cuda.empty_cache()

