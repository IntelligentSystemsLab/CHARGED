# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 16:48
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 16:48
"""
Module for parsing command-line arguments for EV charging demand prediction across multiple cities.

Provides a single function `parse_args` that returns an `argparse.Namespace` of all configuration parameters.
"""
import argparse


def parse_args():
    """
    Parse and return command-line arguments for the EV prediction script.

    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            city (str): City abbreviation (default: 'SPO').
            device (int): CUDA device index (default: 0).
            seed (int): Random seed for reproducibility (default: 2025).
            feature (str): Feature type to predict ('volume' or 'duration').
            auxiliary (str): Auxiliary variable(s) to include (default: 'None').
            data_path (str): Path to input data directory (default: '../data/').
            output_path (str): Directory to save results (default: './result/univariate/').
            model (str): Model name to use for prediction (default: 'convtimenet').
            seq_l (int): Input sequence length (default: 12).
            pre_len (int): Prediction horizon length (default: 1).
            fold (int): Current fold index for cross-validation (default: 1).
            total_fold (int): Total number of folds (default: 6).
            pred_type (str): Granularity of prediction ('station' or 'city').
            batch_size (int): Batch size for training (default: 32).
            epoch (int): Maximum number of training epochs (default: 50).
            is_train (bool): Flag indicating training mode (default: True).
    """
    parser = argparse.ArgumentParser(
        description="EV Charging Demand Prediction across Multiple Cities Worldwide!"
    )

    parser.add_argument(
        '--city', type=str, default='SPO', help="City abbreviation."
    )
    parser.add_argument(
        '--device', type=int, default=0, help="CUDA device index."
    )
    parser.add_argument(
        '--seed', type=int, default=2025, help="Random seed."
    )
    parser.add_argument(
        '--feature', type=str, default='volume', help="Which feature to use for prediction ('volume' or 'duration')."
    )
    parser.add_argument(
        '--auxiliary', type=str, default='None',
        help="Auxiliary variable(s) to include ('None', 'all', or combination)."
    )
    parser.add_argument(
        '--data_path', type=str, default='../data/', help="Path to data directory."
    )
    parser.add_argument(
        '--output_path', type=str, default='./result/univariate/', help="Path to save results."
    )
    parser.add_argument(
        '--model', type=str, default='convtimenet', help="Model name to use."
    )
    parser.add_argument(
        '--seq_l', type=int, default=12, help="Input sequence length."
    )
    parser.add_argument(
        '--pre_len', type=int, default=1, help="Prediction horizon length."
    )
    parser.add_argument(
        '--fold', type=int, default=1, help="Current fold number for cross-validation."
    )
    parser.add_argument(
        '--total_fold', type=int, default=6, help="Total number of folds for cross-validation."
    )
    parser.add_argument(
        '--pred_type', type=str, default='station', help="Prediction granularity ('station' or 'city')."
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        '--epoch', type=int, default=50, help="Maximum training epochs."
    )
    parser.add_argument(
        '--is_train', action='store_true', default=True,
        help="Flag indicating whether to run in training mode."
    )

    return parser.parse_args()
