# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 21:49
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 21:49
"""
Module for parsing command-line arguments for federated EV charging demand prediction across multiple cities.

Provides a single function `federated_parse_args` returning an `argparse.Namespace` with federated learning configuration parameters.
"""
import argparse


def federated_parse_args():
    """
    Parse and return command-line arguments for federated EV prediction.

    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            city (str): Comma-separated list or code for all participating cities (default: 'SZH').
            eval_city (str): City code reserved for evaluation (default: 'SZH').
            max_stations (int): Maximum number of stations per city (default: 200).
            eval_percentage (int): Percentage of stations for evaluation in station mode (default: 20).
            device (int): CUDA device index (default: 0).
            seed (int): Random seed for reproducibility (default: 2025).
            feature (str): Feature type to predict ('volume' or 'duration').
            auxiliary (str): Auxiliary variable(s) to include (default: 'None').
            data_path (str): Path to input data directory (default: '../data/').
            output_path (str): Directory to save results (default: './result/federated/').
            model (str): Model name to use for prediction (default: 'multipatchformer').
            seq_l (int): Input sequence length (default: 12).
            pre_len (int): Prediction horizon length (default: 1).
            pred_type (str): Prediction granularity ('station' or 'city').
            batch_size (int): Batch size for local training (default: 32).
            global_epoch (int): Number of global federated rounds (default: 50).
            local_epoch (int): Number of local training epochs per client (default: 1).
            deploy_epoch (int): Number of deployment/testing epochs during localization (default: 10).
            is_train (bool): Flag indicating training mode (default: True).
    """
    parser = argparse.ArgumentParser(
        description="EV Charging Demand Prediction across Multiple Cities Worldwide with Federated Learning!"
    )

    parser.add_argument(
        '--city', type=str, default='SZH',
        help="All cities' abbreviations or identifier."
    )
    parser.add_argument(
        '--eval_city', type=str, default='SZH',
        help="City abbreviation reserved for evaluation."
    )
    parser.add_argument(
        '--max_stations', type=int, default=200,
        help="Maximum number of stations per city."
    )
    parser.add_argument(
        '--eval_percentage', type=int, default=20,
        help="Percentage of stations for evaluation in station mode."
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help="CUDA device index."
    )
    parser.add_argument(
        '--seed', type=int, default=2025,
        help="Random seed."
    )
    parser.add_argument(
        '--feature', type=str, default='volume',
        help="Which feature to use for prediction ('volume' or 'duration')."
    )
    parser.add_argument(
        '--auxiliary', type=str, default='None',
        help="Which auxiliary variable(s) to include ('None', 'all', or combination)."
    )
    parser.add_argument(
        '--data_path', type=str, default='../data/',
        help="Path to data directory."
    )
    parser.add_argument(
        '--output_path', type=str, default='./result/federated/',
        help="Path to save results."
    )
    parser.add_argument(
        '--model', type=str, default='multipatchformer',
        help="Model name to use."
    )
    parser.add_argument(
        '--seq_l', type=int, default=12,
        help="Input sequence length."
    )
    parser.add_argument(
        '--pre_len', type=int, default=1,
        help="Prediction horizon length."
    )
    parser.add_argument(
        '--pred_type', type=str, default='station',
        help="Prediction granularity ('station' or 'city')."
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help="Batch size for fine-tuning."
    )
    parser.add_argument(
        '--global_epoch', type=int, default=50,
        help="Number of global federated rounds."
    )
    parser.add_argument(
        '--local_epoch', type=int, default=1,
        help="Number of local training epochs per client."
    )
    parser.add_argument(
        '--deploy_epoch', type=int, default=10,
        help="Number of deployment/testing epochs during localization."
    )
    parser.add_argument(
        '--is_train', action='store_true', default=True,
        help="Flag indicating whether to run in training mode."
    )

    return parser.parse_args()
