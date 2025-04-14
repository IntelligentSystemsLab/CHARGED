# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 16:48
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 16:48

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="EV Charging Demand Prediction across Multiple Cities Worldwide!")

    parser.add_argument('--city', type=str, default='SPO', help="City abbreviation.")
    parser.add_argument('--device', type=int, default=0, help="CUDA.")
    parser.add_argument('--seed', type=int, default=2025, help="Random seed.")
    parser.add_argument('--feature', type=str, default='volume', help="Which feature to use for prediction.")
    parser.add_argument('--auxiliary', type=str, default='None', help="Which auxiliary variable to use for prediction.")
    parser.add_argument('--data_path', type=str, default='../data/', help="Path to data.")
    parser.add_argument('--output_path', type=str, default='./result/univariate/', help="Path to save results.")
    parser.add_argument('--model', type=str, default='convtimenet', help="The used model.")
    parser.add_argument('--seq_l', type=int, default=12, help="The sequence length of input data.")
    parser.add_argument('--pre_len', type=int, default=1, help="The length of prediction interval.")
    parser.add_argument('--fold', type=int, default=1, help="The current fold number for training data.")
    parser.add_argument('--total_fold', type=int, default=6, help="The fold used for spliting data in cross-validation")
    parser.add_argument('--pred_type', type=str, default='station', help="What level of prediction.")
    parser.add_argument('--batch_size', type=int, default=32, help="The batch size of fine-tuning.")
    parser.add_argument('--epoch', type=int, default=50, help="The max epoch of the training process.")
    parser.add_argument('--is_train', action='store_true', default=True)

    return parser.parse_args()
