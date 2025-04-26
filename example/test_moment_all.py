# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/04/06 14:13
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/04/06 14:13
"""
Script for evaluating pretrained MOMENT time-series models on EV charging segment-level data.

Loads hourly volume data, aggregates into six daily segments, prepares test tensors,
runs forecasts using MOMENTPipeline, and computes regression metrics.
"""
import json
import os
import torch
import numpy as np
import pandas as pd
from momentfm import MOMENTPipeline

from api.utils import random_seed, calculate_regression_metrics


def main():
    """
    Execute MOMENT-based forecasting evaluation.

    Workflow:
      1. Seed randomness and select compute device.
      2. Load and prepare segment-level test data and labels.
      3. For each model size (small/base/large):
         a) Load pretrained MOMENTPipeline.
         b) Batch predict segment-level outputs.
         c) Reshape and sum to daily forecasts.
         d) Save forecasts and compute regression metrics.
    """

    model_name = 'moment'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    random_seed(2025)

    # Prepare test data
    test_data = []
    test_label = []
    min_val = np.inf
    max_val = 0
    cities = ['SZH']
    for city in cities:
        file_path = f'../data/{city}_remove_zero/volume.csv'
        volume = pd.read_csv(file_path, header=0, index_col=0)
        volume.index = pd.to_datetime(volume.index)
        volume['segment'] = volume.index.hour.map(lambda h: 0 if h < 4 else (1 if h < 8 else (2 if h < 12 else (3 if h < 16 else (4 if h < 20 else 5 )))))
        volume['date'] = volume.index.date
        volume_grouped = volume.groupby(['date', 'segment']).sum()
        volume_grouped = volume_grouped.reset_index()
        volume_grouped['date'] = pd.to_datetime(volume_grouped['date'])
        volume_grouped.set_index(['date', 'segment'], inplace=True)

        volume2 = pd.read_csv(file_path, header=0, index_col=0)
        volume2.index = pd.to_datetime(volume2.index)
        volume2 = volume2.resample('D').sum()

        for station in volume.columns[:-2]:
            series = volume_grouped[station]
            series2 = volume2[station]
            data_series = series[(series.index.get_level_values(0).month >= 4) & (series.index.get_level_values(0).month <= 8)]
            label_series = series2[series2.index.month == 9]
            test_data.append(data_series.values)
            test_label.append(label_series.values)
    test_data = torch.tensor(test_data)[:,-512:]
    test_data=test_data.unsqueeze(dim=1).float()
    test_label = torch.tensor(test_label)

    # Forecast horizon: 180 segments (30 days * 6 segments)
    for name_size in ['small','base','large']:
        model_path=f'{model_name}-1-{name_size}'
        model = MOMENTPipeline.from_pretrained(
            f"path_to_your_model/{model_path}", # AutonLab/{model_path}
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': 180,
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': True,
                'freeze_embedder': True,
                'freeze_head': False,
            },
        )
        model.init()
        model.to(device)

        all_forecasts = []
        batch_size = 256
        num_samples = test_data.size(0)
        for i in range(0, num_samples, batch_size):
            batch = test_data[i:i + batch_size].to(device)
            output = model(x_enc=batch)
            forecast = output.forecast
            all_forecasts.append(forecast.detach().cpu())
        full_forecast = torch.cat(all_forecasts, dim=0)

        # Reshape to [N, 30 days, 6 segments] and sum to daily
        full_forecast=full_forecast.squeeze(dim=1)
        full_forecast_np=np.array(full_forecast)
        full_forecast_np_daily=full_forecast_np.reshape(full_forecast_np.shape[0],30,6)
        full_forecast_np_daily=full_forecast_np_daily.sum(axis=2)

        # Save and evaluate
        output_forecast_path = f'./result/{model_name}/{model_name}-{name_size}_before.npy'
        np.save(output_forecast_path, full_forecast_np_daily)
        result_before=calculate_regression_metrics(np.array(test_label), full_forecast_np_daily)
        with open(f'./result/{model_name}/{model_name}-{name_size}_before.json', 'w', encoding='utf-8') as f:
            json.dump(result_before, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()