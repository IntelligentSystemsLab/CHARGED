# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 16:16
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 16:16

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import os
import sys
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from api.config.parse import parse_args
from api.dataset.config import EVDataset
from api.model.config import PredictionModel
from api.trainer.config import PredictionTrainer
from api.utils import random_seed, get_n_feature

if __name__ == '__main__':
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
    args = parse_args()
    base_path = f'{args.output_path}{args.city}/{args.model}/{args.feature}-{args.auxiliary}-{args.pred_type}-{args.seq_l}-{args.pre_len}-{args.fold}'
    new_path = base_path
    counter = 0
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}#{counter}/"
    os.makedirs(new_path)
    logging.basicConfig(
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        level=logging.DEBUG,
        filename=os.path.join(new_path, 'logging.txt'),
        filemode='a'
    )

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    random_seed(seed=args.seed)
    ev_dataset = EVDataset(
        feature=args.feature,
        auxiliary=args.auxiliary,
        data_path=f'{args.data_path}{args.city}_remove_zero/',
    )
    logging.info(
        f"Running {args.model} with feature={args.feature}, pre_l={args.pre_len}, fold={args.fold}, auxiliary={args.auxiliary}, pred_type(node)={args.pred_type}")
    print(
        f"Running {args.model} with feature={args.feature}, pre_l={args.pre_len}, fold={args.fold}, auxiliary={args.auxiliary}, pred_type(node)={args.pred_type}")
    num_node = ev_dataset.feat.shape[1] if args.pred_type == 'region' else 1
    n_fea = get_n_feature(ev_dataset.extra_feat)
    ev_model = PredictionModel(
        num_node=num_node,
        n_fea=n_fea,
        model_name=args.model,
        seq_l=args.seq_l,
        pre_len=args.pre_len,
        device=device,
    )
    if args.model not in ['lo','ar','arima']:
        ev_model.model=ev_model.model.to(device)

    ev_dataset.split_cross_validation(
        fold=args.fold,
        total_fold=args.total_fold,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VAL_RATIO,
        pred_type=args.pred_type,
    )

    logging.info(
        f"Split outcome - Training set: {len(ev_dataset.train_feat)}, Validation set: {len(ev_dataset.valid_feat)}, Test set: {len(ev_dataset.test_feat)}")

    print(
        f"Split outcome - Training set: {len(ev_dataset.train_feat)}, Validation set: {len(ev_dataset.valid_feat)}, Test set: {len(ev_dataset.test_feat)}")
    if args.city in ['AMS','SZH']:
        args.batch_size=8
        ev_model.update_chunksize(64)
    ev_dataset.create_loaders(
        seq_l=args.seq_l,
        pre_len=args.pre_len,
        batch_size=args.batch_size,
        device=device,
    )

    ev_trainer = PredictionTrainer(
        dataset=ev_dataset,
        model=ev_model,
        seq_l=args.seq_l,
        pre_len=args.pre_len,
        is_train=args.is_train,
        save_path=new_path,
    )

    if ev_trainer.is_train:
        ev_trainer.training(epoch=args.epoch)
    ev_trainer.test()
