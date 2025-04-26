# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 15:26
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 15:26


"""
Main entry point for single-city EV charging demand prediction.

Parses command-line arguments, initializes dataset and model,
performs cross-validation split, trains (if enabled), and tests.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import torch

# Ensure parent directory is in path for package imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from api.parsing.common import parse_args
from api.dataset.common import EVDataset
from api.model.config import PredictionModel
from api.trainer.common import PredictionTrainer
from api.utils import random_seed, get_n_feature, Logger


def main():
    """
    Execute single-city EV demand prediction workflow.

    Steps:
    1. Parse CLI arguments.
    2. Configure output directory and logging.
    3. Seed randomness and select compute device.
    4. Load EVDataset and initialize model.
    5. Perform cross-validation split and DataLoader creation.
    6. Train model if enabled and evaluate on test set.
    """
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
    args = parse_args()
    base_path = f'{args.output_path}{args.city}/{args.model}/{args.feature}-{args.auxiliary}-{args.pred_type}-{args.seq_l}-{args.pre_len}-{args.fold}'
    new_path = base_path
    counter = 0
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}#{counter}/"
    os.makedirs(new_path)
    sys.stdout=Logger(os.path.join(new_path, 'logging.txt'))


    # Device and reproducibility
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    random_seed(args.seed)

    # Load dataset
    data_dir = f"{args.data_path}{args.city}_remove_zero/"
    ev_dataset = EVDataset(
        feature=args.feature,
        auxiliary=args.auxiliary,
        data_path=data_dir,
    )
    print(
        f"Running {args.model} on {args.city} with feature={args.feature}, "
        f"pre_len={args.pre_len}, fold={args.fold}, auxiliary={args.auxiliary}, pred_type={args.pred_type}"
    )

    # Determine number of nodes and features
    num_node = ev_dataset.feat.shape[1] if args.pred_type == 'station' else 1
    n_fea = get_n_feature(ev_dataset.extra_feat)

    # Initialize prediction model
    ev_model = PredictionModel(
        num_node=num_node,
        n_fea=n_fea,
        model_name=args.model,
        seq_l=args.seq_l,
        pre_len=args.pre_len,
    )
    if args.model not in ('lo', 'ar', 'arima'):
        ev_model.model = ev_model.model.to(device)

    # Cross-validation split
    ev_dataset.split_cross_validation(
        fold=args.fold,
        total_fold=args.total_fold,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VAL_RATIO,
    )

    print(
        f"Split outcome - Training set: {len(ev_dataset.train_feat)}, Validation set: {len(ev_dataset.valid_feat)}, Test set: {len(ev_dataset.test_feat)}")
    ev_dataset.create_loaders(
        seq_l=args.seq_l,
        pre_len=args.pre_len,
        batch_size=args.batch_size,
        device=device,
    )

    # Initialize trainer
    ev_trainer = PredictionTrainer(
        dataset=ev_dataset,
        model=ev_model,
        seq_l=args.seq_l,
        pre_len=args.pre_len,
        is_train=args.is_train,
        save_path=new_path,
    )

    # Train and test
    if ev_trainer.is_train:
        ev_trainer.training(epoch=args.epoch)
    ev_trainer.test()


if __name__ == '__main__':
    main()
