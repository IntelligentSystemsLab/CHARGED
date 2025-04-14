# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 21:49
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 21:49

import argparse

def federated_parse_args():
    parser = argparse.ArgumentParser(description="EV Charging Demand Prediction across Multiple Cities Worldwide with Federated Learning!")

    parser.add_argument('--city', type=str, default='SZH', help="All cities' abbreviation.")
    parser.add_argument('--eval_city', type=str, default='SZH', help="Cities' abbreviation for evaluation.")
    parser.add_argument('--max_stations', type=int, default=200, help="Number of max stations.")
    parser.add_argument('--eval_percentage', type=int, default=20, help="Percentage (%) of evaluation stations.")
    parser.add_argument('--device', type=int, default=0, help="CUDA.")
    parser.add_argument('--seed', type=int, default=2025, help="Random seed.")
    parser.add_argument('--feature', type=str, default='volume', help="Which feature to use for prediction.")
    parser.add_argument('--auxiliary', type=str, default='None', help="Which auxiliary variable to use for prediction.")
    parser.add_argument('--data_path', type=str, default='../data/', help="Path to data.")
    parser.add_argument('--output_path', type=str, default='./result/federated/', help="Path to save results.")
    parser.add_argument('--model', type=str, default='multipatchformer', help="The used model.")
    parser.add_argument('--seq_l', type=int, default=12, help="The sequence length of input data.")
    parser.add_argument('--pre_len', type=int, default=1, help="The length of prediction interval.")
    parser.add_argument('--pred_type', type=str, default='station', help="What level of prediction.")
    parser.add_argument('--batch_size', type=int, default=32, help="The batch size of fine-tuning.")
    parser.add_argument('--global_epoch', type=int, default=50, help="The epochs of the global training process.")
    parser.add_argument('--local_epoch', type=int, default=1, help="The epochs of the local training process.")
    parser.add_argument('--deploy_epoch', type=int, default=10, help="The epochs of the model deploy process.")
    parser.add_argument('--is_train', action='store_true', default=True)

    return parser.parse_args()
