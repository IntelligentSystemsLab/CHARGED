# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 20:14
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 20:14
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

from api.utils import get_n_feature

"""
Module for loading and processing electric vehicle dataset features,
including volume or duration, pricing, weather, and station information.
Provides functionality for cross-validation splits, distributed dataset handling,
client partitioning, and DataLoader creation.
"""


class DistributedEVDataset(object):
    """
    Distributed dataset handler for EV features across multiple clients (cities/stations).

    This class loads and preprocesses data per city, scales features,
    constructs auxiliary arrays, partitions data into client datasets,
    and provides methods for retrieving client-specific data.

    Attributes:
        feature (str): Type of feature to load ('volume' or 'duration').
        auxiliary (str): Auxiliary feature modes ('None', 'all', or '+'-separated list).
        clients_data (dict): Mapping client_id to its data dict with keys 'feat', 'extra_feat', 'time'.
        city_scalers (dict): Mapping city codes to their max value scalers.
        n_fea (int): Number of feature channels in extra_feat.
        training_clients_data (dict): Clients reserved for training.
        eval_clients_data (dict): Clients reserved for evaluation.
    """

    def __init__(
            self,
            feature: str,
            auxiliary: str,
            data_paths: dict[str, str],
            pred_type: str,
            eval_percentage: float,
            eval_city: str,
            max_stations: int = 300,
            weather_columns: list[str] = ['temp', 'precip', 'visibility'],
            selection_mode: str = 'middle',
    ) -> None:
        """
        Initialize DistributedEVDataset by loading and preprocessing per-city data.

        Args:
            feature (str): Feature file to load; 'volume' or 'duration'.
            auxiliary (str): Auxiliary data mode: 'None', 'all', or combination of features.
            data_paths (dict): Mapping city abbreviations to data directories.
            pred_type (str): Prediction granularity; 'station' or 'city'.
            eval_percentage (float): % of clients per city for evaluation (station mode).
            eval_city (str): City code reserved for evaluation (city mode).
            max_stations (int): Max stations per city (station mode).
            weather_columns (list[str]): Weather attributes to include.
            selection_mode (str): Station selection strategy: 'top', 'middle', 'random'.

        Raises:
            ValueError: If feature or pred_type or selection_mode is invalid.
        """
        super(DistributedEVDataset, self).__init__()
        self.feature = feature
        self.auxiliary = auxiliary
        self.clients_data = {}
        self.city_scalers = {}

        # Process each city's data
        for city_abbr, data_path in data_paths.items():
            # Load and normalize main feature DataFrame
            if self.feature == 'volume':
                feat_df = pd.read_csv(f'{data_path}volume.csv', header=0, index_col=0)
            elif self.feature == 'duration':
                feat_df = pd.read_csv(f'{data_path}duration.csv', header=0, index_col=0)
            else:
                raise ValueError("Unknown feature")

            # Scale by city max
            max_val = feat_df.max().max()
            feat_scaled = feat_df / max_val
            self.city_scalers[city_abbr] = max_val
            feat_df = feat_scaled.copy()

            station_ids = list(feat_df.columns)

            # Load and normalize price data
            e_price_all = pd.read_csv(f'{data_path}e_price.csv', index_col=0, header=0).values
            s_price_all = pd.read_csv(f'{data_path}s_price.csv', index_col=0, header=0).values
            time_series = pd.to_datetime(feat_df.index)
            price_scaler = MinMaxScaler(feature_range=(0, 1))
            e_price_all = price_scaler.fit_transform(e_price_all)
            s_price_all = price_scaler.fit_transform(s_price_all)

            # Load and normalize weather data
            weather = pd.read_csv(f'{data_path}weather.csv', header=0, index_col='time')
            weather = weather[weather_columns]
            if 'temp' in weather.columns:
                weather['temp'] = (weather['temp'] + 5) / 45
            if 'precip' in weather.columns:
                weather['precip'] = weather['precip'] / 120
            if 'visibility' in weather.columns:
                weather['visibility'] = weather['visibility'] / 50

            # Load station metadata
            stations_info = pd.read_csv(f'{data_path}stations.csv').set_index('station_id')
            stations_info.index = stations_info.index.astype(str)

            # Station selection if pred_type == 'station'
            if pred_type == 'station' and len(stations_info) > max_stations:
                if selection_mode == 'top':
                    selected_stations = stations_info.sort_values(by='total_duration', ascending=False).head(
                        max_stations)
                elif selection_mode == 'middle':
                    sorted_stations = stations_info.sort_values(by='total_duration', ascending=True)
                    start = max((len(sorted_stations) - max_stations) // 2, 0)
                    selected_stations = sorted_stations.iloc[start:start + max_stations]
                elif selection_mode == 'random':
                    selected_stations = stations_info.sample(n=max_stations, random_state=42)
                else:
                    raise ValueError(f"Unknown selection_mode: {selection_mode}")
                selected_ids = selected_stations.index.tolist()
                feat_df = feat_df[selected_ids]
                e_price_df = pd.read_csv(f'{data_path}e_price.csv', index_col=0, header=0)
                s_price_df = pd.read_csv(f'{data_path}s_price.csv', index_col=0, header=0)
                e_price_all = price_scaler.fit_transform(e_price_df[selected_ids])
                s_price_all = price_scaler.fit_transform(s_price_df[selected_ids])
                stations_info = selected_stations
                station_ids = selected_ids

            # Normalize lat/long and create base extra_feat
            lat_long = stations_info.loc[feat_df.columns, ['latitude', 'longitude']].values
            lat_norm = (lat_long[:, 0] + 90) / 180
            lon_norm = (lat_long[:, 1] + 180) / 360
            lat_long_norm = np.stack([lat_norm, lon_norm], axis=1)
            extra_feat = np.tile(lat_long_norm[np.newaxis, :, :], (feat_df.shape[0], 1, 1))

            # Add auxiliary channels if requested
            if self.auxiliary != 'None':
                extra_feat = np.zeros([feat_df.shape[0], feat_df.shape[1], 1])
                if self.auxiliary == 'all':
                    extra_feat = np.concatenate([extra_feat, e_price_all[:, :, np.newaxis]], axis=2)
                    extra_feat = np.concatenate([extra_feat, s_price_all[:, :, np.newaxis]], axis=2)
                    extra_feat = np.concatenate([extra_feat,
                                                 np.repeat(weather.values[:, np.newaxis, :], feat_df.shape[1], axis=1)],
                                                axis=2)
                else:
                    add_feat_list = self.auxiliary.split('+')
                    for add_feat in add_feat_list:
                        if add_feat == 'e_price':
                            extra_feat = np.concatenate([extra_feat, e_price_all[:, :, np.newaxis]], axis=2)
                        elif add_feat == 's_price':
                            extra_feat = np.concatenate([extra_feat, s_price_all[:, :, np.newaxis]], axis=2)
                        else:
                            extra_feat = np.concatenate([extra_feat,
                                                         np.repeat(weather[add_feat].values[:, np.newaxis, np.newaxis],
                                                                   feat_df.shape[1], axis=1)], axis=2)
                extra_feat = extra_feat[:, :, 1:]

            # Convert to numpy and compute n_fea
            feat_array = np.array(feat_df)
            self.n_fea = get_n_feature(extra_feat)

            # Partition into clients
            if pred_type == 'station':
                for idx, station_id in enumerate(station_ids):
                    client_feat = feat_array[:, idx:idx + 1]
                    client_extra = extra_feat[:, idx:idx + 1, :] if extra_feat is not None else None
                    client_id = f"{city_abbr}_{station_id}"
                    self.clients_data[client_id] = {
                        'feat'      : client_feat,
                        'extra_feat': client_extra,
                        'time'      : time_series
                    }
            elif pred_type == 'city':
                aggregated_feat = np.sum(feat_array, axis=1, keepdims=True)
                aggregated_extra = np.mean(extra_feat, axis=1, keepdims=True) if extra_feat is not None else None
                self.clients_data[city_abbr] = {
                    'feat'      : aggregated_feat,
                    'extra_feat': aggregated_extra,
                    'time'      : time_series
                }
            else:
                raise ValueError("Unknown pred_type")

        # Final partition
        self.partition_clients(eval_percentage, eval_city, pred_type)

    def partition_clients(
            self,
            eval_percentage: float,
            eval_city: str,
            pred_type: str,
    ) -> None:
        """
        Partition client IDs into training and evaluation sets.

        Args:
            eval_percentage (float): Percentage of clients for eval (station mode).
            eval_city (str): City code for eval (city mode).
            pred_type (str): 'station' or 'city'.

        Raises:
            ValueError: If pred_type is invalid.
        """
        if pred_type == 'station':
            training_clients_data = {}
            eval_clients_data = {}
            city_clients = defaultdict(list)
            for client_id in self.clients_data.keys():
                city = client_id.split('_')[0]
                city_clients[city].append(client_id)
            for city, client_ids in city_clients.items():
                client_ids_sorted = sorted(client_ids)
                num_clients = len(client_ids_sorted)
                num_eval = int(num_clients * eval_percentage / 100)
                if num_clients > 0 and num_eval == 0 and eval_percentage > 0:
                    num_eval = 1
                eval_ids = client_ids_sorted[:num_eval]
                train_ids = client_ids_sorted[num_eval:]
                for cid in train_ids:
                    training_clients_data[cid] = self.clients_data[cid]
                for cid in eval_ids:
                    eval_clients_data[cid] = self.clients_data[cid]
            self.training_clients_data = training_clients_data
            self.eval_clients_data = eval_clients_data
        elif pred_type == 'city':
            training_clients_data = {}
            eval_clients_data = {}
            for city, data in self.clients_data.items():
                if city == eval_city:
                    eval_clients_data[city] = data
                else:
                    training_clients_data[city] = data
            self.training_clients_data = training_clients_data
            self.eval_clients_data = eval_clients_data
        else:
            raise ValueError("Unknown pred_type")

    def get_client_ids(self) -> list[str]:
        """
        Get list of all client IDs.

        Returns:
            list[str]: Client identifiers.
        """
        return list(self.clients_data.keys())

    def get_client_data(self, client_id: str) -> dict | None:
        """
        Retrieve data for a specific client.

        Args:
            client_id (str): Identifier of the client.

        Returns:
            dict or None: Data dict with 'feat', 'extra_feat', 'time', or None if missing.
        """
        return self.clients_data.get(client_id, None)
