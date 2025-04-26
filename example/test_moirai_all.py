# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/04/06 14:13
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/04/06 14:13
"""
Script for evaluating pretrained Moirai time-series forecasting models on EV charging demand data.

Loads daily aggregated volume data, prepares GluonTS test instances, runs forecasts
using MoiraiForecast, and computes regression metrics.
"""
import json
import torch
import numpy as np
import pandas as pd

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from api.utils import random_seed, calculate_regression_metrics

def main():
    """
    Execute Moirai-based forecasting workflow.

    Workflow:
      1. Seed randomness and select compute device.
      2. Load daily volume data for specified cities (Apr–Sep).
      3. Construct GluonTS PandasDataset and split into train/test with sliding window.
      4. Generate test instances for prediction.
      5. For each model size (small/base/large):
         a. Load pretrained MoiraiModule.
         b. Create predictor and run forecasts.
         c. Extract mean forecasts and compute regression metrics.
         d. Save forecasts and metrics to disk.
    """
    model_name = 'moirai'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    random_seed(2025)

    # Load and combine data for cities
    cities = ['SZH']
    dfs = []
    for city in cities:
        file_path = f'../data/{city}_remove_zero/volume.csv'
        volume = pd.read_csv(file_path, header=0, index_col=0)
        volume.index = pd.to_datetime(volume.index)
        volume = volume.resample('D').sum()
        volume = volume[(volume.index.month >= 4) & (volume.index.month <= 9)]
        dfs.append(volume)

    # Create GluonTS dataset and split
    combined_df = pd.concat(dfs, axis=1)
    combined_df.columns = [f'station_{i}' for i in range(combined_df.shape[1])]
    ds = PandasDataset(dict(combined_df))
    offset = -30
    train_ds, test_template = split(ds, offset=offset)
    test_instances = test_template.generate_instances(
        prediction_length=30,
        windows=1,
        distance=30
    )

    context_length=153
    prediction_length = 30

    # Forecast with different model sizes
    for name_size in ['small','base','large']:
        test_label = []
        model_path=f'{model_name}-1.1-R-{name_size}'
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"path_to_your_model/{model_path}"), # Salesforce/{model_path}
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size="auto",
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
        predictor = model.create_predictor(batch_size=32)
        forecasts = predictor.predict(test_instances.input)

        # Collect true labels
        for l in test_instances.label:
            test_label.append(l['target'])

        # Predict and average
        forecast_all = []
        for f in forecasts:
            forecast_all.append(f.mean)
        forecast_mean_np = np.array(forecast_all)

        # Save and evaluate
        output_forecast_path = f'./result/{model_name}/{model_name}-{name_size}_before.npy'
        np.save(output_forecast_path, forecast_mean_np)
        result_before=calculate_regression_metrics(np.array(test_label), forecast_mean_np)
        with open(f'./result/{model_name}/{model_name}-{name_size}_before.json', 'w', encoding='utf-8') as f:
            json.dump(result_before, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
