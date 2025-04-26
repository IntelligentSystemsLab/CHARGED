# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/04/06 14:13
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/04/06 14:13
"""
Script for evaluating pretrained Chronos T5 models on EV charging demand data.

Provides utilities to normalize and tokenize time series, then loads daily
aggregated volume data, runs ChronosPipeline forecasts, and computes regression metrics.
"""
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from chronos import ChronosPipeline

from api.utils import random_seed, calculate_regression_metrics


def main():
    """
    Load daily EV volume data, run forecasts using pretrained Chronos T5 models,
    and save predictions and regression metrics for different model sizes.

    Workflow:
      1. Seed randomness and select compute device.
      2. Load and aggregate daily volume series for specified cities.
      3. Split into test inputs (Apr–Aug) and labels (September).
      4. For each model size (small/base/large):
         a) Load pretrained ChronosPipeline.
         b) Batch predict next 30 days.
         c) Compute mean forecast across ensembles.
         d) Save forecast and compute regression metrics.
    """
    model_name = 'chronos'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    random_seed(2025)

    test_data, test_label = [], []
    cities = ['SZH']

    # Load and prepare data
    for city in cities:
        file_path = f'../data/{city}_remove_zero/volume.csv'
        volume = pd.read_csv(file_path, header=0, index_col=0)
        volume.index = pd.to_datetime(volume.index)
        volume = volume.resample('D').sum()
        for station in volume.columns:
            series = volume[station]
            data_series = series[(series.index.month >= 4) & (series.index.month <= 8)]
            label_series = series[series.index.month == 9]
            test_data.append(data_series.values)
            test_label.append(label_series.values)
    test_data = torch.tensor(test_data)
    test_label = torch.tensor(test_label)

    # Forecast with different model sizes
    for size in ['small', 'base', 'large']:
        checkpoint = f'path_to_your_model/{model_name}-t5-{size}'
        pipeline = ChronosPipeline.from_pretrained(
            checkpoint,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
        )

        prediction_length = 30
        batch_size = 32
        num_batches = (test_data.shape[0] + batch_size - 1) // batch_size

        forecast_all = []

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, test_data.shape[0])
            batch_data = test_data[start_idx:end_idx]
            forecast_batch = pipeline.predict(batch_data, prediction_length=prediction_length)
            forecast_all.append(forecast_batch)

        forecast = torch.cat(forecast_all, dim=0)
        forecast_mean = torch.mean(forecast, dim=1)
        forecast_mean_np = forecast_mean.cpu().numpy()

        # Save and evaluate
        output_forecast_path = f'./result/{model_name}/{model_name}-{size}_before.npy'
        np.save(output_forecast_path, forecast_mean_np)
        result_before = calculate_regression_metrics(np.array(test_label), forecast_mean_np)
        with open(f'./result/{model_name}/{model_name}-{size}_before.json', 'w', encoding='utf-8') as f:
            json.dump(result_before, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()