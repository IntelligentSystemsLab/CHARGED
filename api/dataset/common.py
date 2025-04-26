# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 17:16
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 17:16

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

from api.utils import CreateDataset

"""
Module for loading and processing electric vehicle dataset features,
including volume or duration, pricing, weather, and station information.
Provides functionality for cross-validation splits and DataLoader creation.
"""


class EVDataset(object):
    """
    Dataset handler for EV (electric vehicle) features with auxiliary data.

    This class loads feature data (volume or duration), price data, weather,
    and station metadata, normalizes and selects stations, constructs
    auxiliary features, and provides methods for cross-validation splitting
    and DataLoader creation.

    Attributes:
        feature (str): Type of feature to load ('volume' or 'duration').
        auxiliary (str): Auxiliary feature modes ('None', 'all', or '+'-separated list).
        data_path (str): Root path to CSV files.
        feat (np.ndarray): Primary feature array [time, stations].
        extra_feat (np.ndarray): Auxiliary feature array [time, stations, features].
        time (pd.DatetimeIndex): Timestamps for each row in feat.
        scaler (StandardScaler): Scaler fitted on training data.
        train_feat, valid_feat, test_feat (np.ndarray): Splits of feat.
        train_extra_feat, valid_extra_feat, test_extra_feat (np.ndarray): Splits of extra_feat.
        train_loader, valid_loader, test_loader (DataLoader): PyTorch DataLoaders.
    """

    def __init__(
            self,
            feature: str,
            auxiliary: str,
            data_path: str,
            max_stations: int = 300,
            weather_columns: list[str] = ['temp', 'precip', 'visibility'],
            selection_mode: str = 'top',
    ) -> None:
        """
        Initialize EVDataset by loading and preprocessing data.

        Args:
            feature (str): Feature file to load; 'volume' or 'duration'.
            auxiliary (str): Auxiliary data mode: 'None', 'all', or combination of 'e_price', 's_price', weather keys.
            data_path (str): Directory path containing CSV files.
            max_stations (int): Maximum number of stations to select.
            weather_columns (list[str]): Columns to load from weather data.
            selection_mode (str): Station selection mode: 'top', 'middle', or 'random'.

        Raises:
            ValueError: If feature name or selection_mode is invalid.
        """
        super(EVDataset, self).__init__()
        self.feature = feature
        self.auxiliary = auxiliary
        self.data_path = data_path

        # Load main feature DataFrame based on feature type
        if self.feature == 'volume':
            self.feat = pd.read_csv(f'{self.data_path}volume.csv', header=0, index_col=0)
        elif self.feature == 'duration':
            self.feat = pd.read_csv(f'{self.data_path}duration.csv', header=0, index_col=0)
        else:
            raise ValueError("Unknown feature")

        # Load and normalize price data
        self.e_price = pd.read_csv(f'{self.data_path}e_price.csv', index_col=0, header=0).values
        self.s_price = pd.read_csv(f'{self.data_path}s_price.csv', index_col=0, header=0).values
        self.time = pd.to_datetime(self.feat.index)

        price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.e_price = price_scaler.fit_transform(self.e_price)
        self.s_price = price_scaler.fit_transform(self.s_price)

        # Load and normalize weather data
        self.weather = pd.read_csv(f'{self.data_path}weather.csv', header=0, index_col='time')
        self.weather = self.weather[weather_columns]
        if 'temp' in self.weather.columns:
            # Scale temperature to [0,1]
            self.weather['temp'] = (self.weather['temp'] + 5) / 45
        if 'precip' in self.weather.columns:
            # Scale precipitation to [0,1]
            self.weather['precip'] = self.weather['precip'] / 120
        if 'visibility' in self.weather.columns:
            # Scale visibility to [0,1]
            self.weather['visibility'] = self.weather['visibility'] / 50

        # Load station metadata
        stations_info = pd.read_csv(f'{self.data_path}stations.csv', header=0)
        stations_info = stations_info.set_index("station_id")
        stations_info.index = stations_info.index.astype(str)

        # Select subset of stations if exceeding max_stations
        if len(stations_info) > max_stations:
            if selection_mode == 'top':
                # Top stations by total_duration
                selected_stations = stations_info.sort_values(
                    by='total_duration', ascending=False
                ).head(max_stations)
            elif selection_mode == 'middle':
                # Middle-range stations by total_duration
                sorted_stations = stations_info.sort_values(
                    by='total_duration', ascending=True
                )
                start = max((len(sorted_stations) - max_stations) // 2, 0)
                selected_stations = sorted_stations.iloc[start:start + max_stations]
            elif selection_mode == 'random':
                # Random sample of stations
                selected_stations = stations_info.sample(
                    n=max_stations, random_state=42
                )
            else:
                raise ValueError(f"Unknown selection_mode: {selection_mode}")

            selected_ids = selected_stations.index.tolist()
            # Filter features and re-scale prices for selected stations
            self.feat = self.feat[selected_ids]
            e_price_df = pd.read_csv(
                f'{self.data_path}e_price.csv', index_col=0, header=0
            )
            s_price_df = pd.read_csv(
                f'{self.data_path}s_price.csv', index_col=0, header=0
            )
            self.e_price = price_scaler.fit_transform(e_price_df[selected_ids])
            self.s_price = price_scaler.fit_transform(s_price_df[selected_ids])

        # Normalize station latitude and longitude to [0,1]
        lat_long = stations_info.loc[self.feat.columns, ['latitude', 'longitude']].values
        lat_norm = (lat_long[:, 0] + 90) / 180
        lon_norm = (lat_long[:, 1] + 180) / 360
        self.lat_long_norm = np.stack([lat_norm, lon_norm], axis=1)
        # Repeat coordinates for each time step
        self.extra_feat = np.tile(
            self.lat_long_norm[np.newaxis, :, :], (self.feat.shape[0], 1, 1)
        )

        # Build auxiliary features if requested
        if self.auxiliary != 'None':
            # Start with zero channel for base
            self.extra_feat = np.zeros([
                self.feat.shape[0], self.feat.shape[1], 1
            ])
            if self.auxiliary == 'all':
                # Add all auxiliary: e_price, s_price, weather
                self.extra_feat = np.concatenate([
                    self.extra_feat, self.e_price[:, :, np.newaxis]
                ], axis=2)
                self.extra_feat = np.concatenate([
                    self.extra_feat, self.s_price[:, :, np.newaxis]
                ], axis=2)
                self.extra_feat = np.concatenate([
                    self.extra_feat,
                    np.repeat(
                        self.weather.values[:, np.newaxis, :],
                        self.feat.shape[1], axis=1
                    )
                ], axis=2)
            else:
                # Add specified auxiliary features
                for add_feat in self.auxiliary.split('+'):
                    if add_feat == 'e_price':
                        self.extra_feat = np.concatenate([
                            self.extra_feat, self.e_price[:, :, np.newaxis]
                        ], axis=2)
                    elif add_feat == 's_price':
                        self.extra_feat = np.concatenate([
                            self.extra_feat, self.s_price[:, :, np.newaxis]
                        ], axis=2)
                    else:
                        # Weather feature channel
                        self.extra_feat = np.concatenate([
                            self.extra_feat,
                            np.repeat(
                                self.weather[add_feat].values[:, np.newaxis, np.newaxis],
                                self.feat.shape[1], axis=1
                            )
                        ], axis=2)
            # Remove initial zero channel
            self.extra_feat = self.extra_feat[:, :, 1:]

        # Convert main feature DataFrame to numpy array
        self.feat = np.array(self.feat)

    def split_cross_validation(
            self,
            fold: int,
            total_fold: int,
            train_ratio: float,
            valid_ratio: float,
    ) -> None:
        """
        Split data by month for cross-validation.

        Args:
            fold (int): Current fold index (1-based).
            total_fold (int): Total number of folds (should equal number of months).
            train_ratio (float): Fraction of fold data for training.
            valid_ratio (float): Fraction of fold data for validation.

        Raises:
            AssertionError: If time and feature lengths mismatch or train set empty.
        """
        # Ensure time length matches data length
        assert len(self.time) == len(self.feat)
        month_list = sorted(np.unique(self.time.month))
        assert total_fold == len(month_list)

        # Determine index split points based on month membership
        fold_time = self.time.month.isin(month_list[0:fold]).sum()
        train_end = int(fold_time * train_ratio)
        valid_start = train_end
        valid_end = int(valid_start + fold_time * valid_ratio)
        test_start = valid_end
        test_end = int(fold_time)

        # Slice numpy arrays for each split
        train_feat = self.feat[:train_end]
        valid_feat = self.feat[valid_start:valid_end]
        test_feat = self.feat[test_start:test_end]

        # Apply standard scaling
        self.scaler = StandardScaler()
        self.train_feat = self.scaler.fit_transform(train_feat)
        self.valid_feat = self.scaler.transform(valid_feat)
        self.test_feat = self.scaler.transform(test_feat)

        self.train_extra_feat, self.valid_extra_feat, self.test_extra_feat = None, None, None
        # Split auxiliary features if present
        if self.extra_feat is not None:
            self.train_extra_feat = self.extra_feat[:train_end]
            self.valid_extra_feat = self.extra_feat[valid_start:valid_end]
            self.test_extra_feat = self.extra_feat[test_start:test_end]

        assert len(train_feat) > 0, "The training set cannot be empty!"

    def create_loaders(
            self,
            seq_l: int,
            pre_len: int,
            batch_size: int,
            device: torch.device,
    ) -> None:
        """
        Create PyTorch DataLoaders for training, validation, and testing.

        Args:
            seq_l (int): Input sequence length.
            pre_len (int): Prediction horizon length.
            batch_size (int): Batch size for training loader.
            device (torch.device): Device for tensor allocation.
        """
        # Build dataset objects
        train_dataset = CreateDataset(
            seq_l, pre_len, self.train_feat, self.train_extra_feat, device
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        valid_dataset = CreateDataset(
            seq_l, pre_len, self.valid_feat, self.valid_extra_feat, device
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=len(self.valid_feat), shuffle=False
        )

        test_dataset = CreateDataset(
            seq_l, pre_len, self.test_feat, self.test_extra_feat, device
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=len(self.test_feat), shuffle=False
        )
