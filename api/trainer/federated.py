# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/15 1:33
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/15 1:33

"""
Module defining ClientTrainer for per-client training and evaluation in federated learning.

Classes:
    ClientTrainer: Manages local training, testing, model checkpointing, and metric logging for a single client.
"""
import json
import os
from typing import Optional, Any, Generator

import numpy as np
import torch
from tqdm import tqdm

from api.utils import calculate_regression_metrics, convert_numpy

"""
Module defining ClientTrainer for per-client training and evaluation in federated learning.

Classes:
    ClientTrainer: Manages local training, testing, model checkpointing, and metric logging for a single client.
"""


class ClientTrainer(object):
    """
    Trainer for a single federated client.

    Handles training, evaluation, checkpointing, and metrics for neural prediction models.

    Attributes:
        train_loader (DataLoader): Local training data loader.
        test_loader (DataLoader): Local test data loader.
        extra_feat_tag (bool): Indicates if auxiliary features are used.
        ev_model (Any): PredictionModel instance with `.model` and `.model_name`.
        save_path (str): Directory to save model and results.
        scaler (float | np.ndarray): Scale factor for inverse transformation.
        device (torch.device): Device for computation.
        optim (Optimizer): Optimizer for model training.
        loss_func (torch.nn.Module): Loss criterion.
        now_epoch (int): Current global federated epoch.
        deploy_epoch (int): Current localization epoch.
    """

    def __init__(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            extra_feat_tag: bool,
            model: Any,
            save_path: str,
            scaler: float | np.ndarray,
            device: torch.device,
    ) -> None:
        """
        Initialize ClientTrainer.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for testing data.
            extra_feat_tag (bool): True if using auxiliary features.
            model (Any): PredictionModel with `.model_name` and `.model`.
            save_path (str): Path to save checkpoints and outputs.
            scaler (float | np.ndarray): Scale for inverse-transform.
            device (torch.device): Device for training/testing.

        Raises:
            ValueError: If statistical models are used.
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.extra_feat_tag = extra_feat_tag
        self.ev_model = model
        self.save_path = save_path
        self.device = device
        self.scaler = scaler
        self.now_epoch = 0
        self.deploy_epoch = 0

        if model.model_name in ('lo', 'ar', 'arima'):
            raise ValueError("ClientTrainer is not applicable to statistical models.")

        self.optim = torch.optim.Adam(
            model.model.parameters(), weight_decay=1e-5
        )
        self.loss_func = torch.nn.MSELoss()

    def training(
            self,
            epoch: int,
            save_model: bool = False
    ) -> None:
        """
        Perform local training for a specified number of epochs.

        Args:
            epoch (int): Number of local epochs to train.
            save_model (bool): Whether to save model checkpoint after training.
        """
        self.ev_model.model.train()
        self.ev_model.model.to(self.device)

        for _ in tqdm(range(epoch), desc='Training'):
            for feat, label, extra_feat in self.train_loader:
                torch.cuda.empty_cache()
                if not self.extra_feat_tag:
                    extra_feat = None
                self.optim.zero_grad()
                preds = self.ev_model.model(feat, extra_feat)
                # Align shapes for loss computation
                if preds.shape != label.shape:
                    loss = self.loss_func(preds.unsqueeze(-1), label)
                else:
                    loss = self.loss_func(preds, label)
                loss.backward()
                self.optim.step()

        if save_model:
            os.makedirs(self.save_path, exist_ok=True)
            path = os.path.join(
                self.save_path,
                f'train_{self.now_epoch}_{self.deploy_epoch}.pth'
            )
            torch.save(self.ev_model.model.state_dict(), path)

        # Move model to CPU and clear GPU cache
        self.ev_model.model.to('cpu')
        torch.cuda.empty_cache()

    def test(
            self,
            model_path: Optional[str] = None
    ) -> None:
        """
        Evaluate the model on the local test set and save predictions and metrics.

        Args:
            model_path (str, optional): Path to a saved model checkpoint to load before testing.
        """
        preds_list, labels_list = [], []
        self.ev_model.model.to(self.device)

        if model_path:
            self.ev_model.load_model(model_path=model_path)
        self.ev_model.model.eval()

        for feat, label, extra_feat in self.test_loader:
            torch.cuda.empty_cache()
            if not self.extra_feat_tag:
                extra_feat = None
            with torch.no_grad():
                preds = self.ev_model.model(feat, extra_feat)
                if preds.shape != label.shape:
                    preds = preds.unsqueeze(-1)
                preds = preds.cpu().numpy()
            labels = label.cpu().numpy()
            preds_list.append(preds)
            labels_list.append(labels)

        # Concatenate and inverse scale
        pred_array = np.concatenate(preds_list, axis=0)
        label_array = np.concatenate(labels_list, axis=0)
        if self.scaler is not None:
            pred_array = pred_array * self.scaler
            label_array = label_array * self.scaler

        # Save outputs
        os.makedirs(self.save_path, exist_ok=True)
        np.save(
            os.path.join(
                self.save_path,
                f'predict_{self.now_epoch}_{self.deploy_epoch}.npy'
            ),
            pred_array
        )
        np.save(
            os.path.join(
                self.save_path,
                f'label_{self.now_epoch}_{self.deploy_epoch}.npy'
            ),
            label_array
        )

        # Compute and save metrics
        metrics = calculate_regression_metrics(y_true=label_array, y_pred=pred_array)
        metrics = convert_numpy(metrics)
        with open(
                os.path.join(
                    self.save_path,
                    f'metrics_{self.now_epoch}_{self.deploy_epoch}.json'
                ), 'w', encoding='utf-8'
        ) as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        # Cleanup
        self.ev_model.model.to('cpu')
        torch.cuda.empty_cache()
