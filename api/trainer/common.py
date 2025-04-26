# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/11 2:56
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/11 2:56
import json
import os
from typing import Optional, Any

import numpy as np
import torch
from tqdm import tqdm

from api.utils import calculate_regression_metrics, convert_numpy

"""
Module defining PredictionTrainer for training and evaluating prediction models.

Classes:
    PredictionTrainer: Manages training loops, validation, testing, and metric logging.
"""


class PredictionTrainer(object):
    """
    Trainer class for prediction models on EV charging demand data.

    Handles both statistical (LO, AR, ARIMA) and neural models, providing
    methods to train, validate, save best model, and evaluate with metrics.

    Attributes:
        ev_dataset (Any): Dataset object containing DataLoaders or raw arrays.
        ev_model (Any): PredictionModel instance with `.model` and `.model_name`.
        is_train (bool): Flag indicating training mode.
        stat_model (bool): True if using statistical model (no optimizer).
        optim (Optional[torch.optim.Optimizer]): Optimizer for neural models.
        loss_func (Optional[torch.nn.Module]): Loss function for neural models.
        save_path (str): Directory to save checkpoints and outputs.
        test_loader (Any): Raw arrays or DataLoader for test data.
        train_valid_feat (Optional[np.ndarray]): Combined data for statistical models.
    """

    def __init__(
            self,
            dataset: Any,
            model: Any,
            seq_l: int,
            pre_len: int,
            is_train: bool,
            save_path: str,
    ) -> None:
        """
        Initialize PredictionTrainer.

        Args:
            dataset (Any): EVDataset or similar with train_loader, valid_loader, test_loader.
            model (Any): PredictionModel containing `.model` and `.model_name`.
            seq_l (int): Input sequence length for statistical models.
            pre_len (int): Prediction horizon length for statistical models.
            is_train (bool): Flag to enable training mode.
            save_path (str): Directory path to save model and results.
        """
        self.ev_dataset = dataset
        self.ev_model = model
        self.save_path = save_path
        self.is_train = is_train

        # Configure based on model type
        if model.model_name in ('lo', 'ar', 'arima'):
            # Statistical models: no optimizer or loss
            self.optim = None
            self.loss_func = None
            self.is_train = False
            self.stat_model = True
            # Prepare train+valid for prediction
            self.train_valid_feat = np.vstack(
                (dataset.train_feat, dataset.valid_feat,
                 dataset.test_feat[: seq_l + pre_len, :])
            )
            # Test features and labels for statistical prediction
            self.test_loader = [
                self.train_valid_feat,
                dataset.test_feat[pre_len + seq_l:, :]
            ]
        else:
            # Neural models: set optimizer and loss
            self.optim = torch.optim.Adam(
                model.model.parameters(), weight_decay=1e-5
            )
            self.stat_model = False
            self.loss_func = torch.nn.MSELoss()

    def training(self, epoch: int) -> None:
        """
        Train and validate neural model over multiple epochs.

        Saves the best model based on validation loss at 'train.pth'.

        Args:
            epoch (int): Number of training epochs to run.
        """
        best_val_loss = float('inf')
        self.ev_model.model.train()

        for _ in tqdm(range(epoch), desc='Training'):
            # Training loop
            for feat, label, extra in self.ev_dataset.train_loader:
                torch.cuda.empty_cache()
                if self.ev_dataset.extra_feat is None:
                    extra = None
                self.optim.zero_grad()
                preds = self.ev_model.model(feat, extra)
                # Align shapes for loss
                if preds.shape != label.shape:
                    loss = self.loss_func(preds.unsqueeze(-1), label)
                else:
                    loss = self.loss_func(preds, label)
                loss.backward()
                self.optim.step()

            # Validation loop
            for feat, label, extra in self.ev_dataset.valid_loader:
                torch.cuda.empty_cache()
                if self.ev_dataset.extra_feat is None:
                    extra = None
                preds = self.ev_model.model(feat, extra)
                if preds.shape != label.shape:
                    loss = self.loss_func(preds.unsqueeze(-1), label)
                else:
                    loss = self.loss_func(preds, label)
                val_loss = loss.item()
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(self.save_path, exist_ok=True)
                    torch.save(
                        self.ev_model.model.state_dict(),
                        os.path.join(self.save_path, 'train.pth')
                    )

    def test(self, model_path: Optional[str] = None) -> None:
        """
        Evaluate model on test data and save predictions, labels, and metrics.

        For neural models, loads saved checkpoint if provided.
        For statistical models, runs .predict on raw arrays.

        Args:
            model_path (str, optional): Path to model weights for neural models.
        """
        preds_list, labels_list = [], []

        if not self.stat_model:
            # Neural model evaluation
            if model_path:
                self.ev_model.load_model(model_path)
            self.ev_model.model.eval()

            for feat, label, extra in self.ev_dataset.test_loader:
                torch.cuda.empty_cache()
                if self.ev_dataset.extra_feat is None:
                    extra = None
                with torch.no_grad():
                    preds = self.ev_model.model(feat, extra)
                    preds = preds.unsqueeze(-1) if preds.shape != label.shape else preds
                    preds = preds.cpu().numpy()
                labels = label.cpu().numpy()
                preds_list.append(preds)
                labels_list.append(labels)
        else:
            # Statistical model prediction
            train_valid, test_feats = self.test_loader
            preds = self.ev_model.model.predict(train_valid, test_feats)
            preds_list.append(preds)
            labels_list.append(test_feats)

        # Concatenate and inverse-transform
        pred_array = np.concatenate(preds_list, axis=0)
        label_array = np.concatenate(labels_list, axis=0)
        if self.ev_dataset.scaler:
            pred_array = self.ev_dataset.scaler.inverse_transform(pred_array)
            label_array = self.ev_dataset.scaler.inverse_transform(label_array)

        # Save outputs
        os.makedirs(self.save_path, exist_ok=True)
        np.save(os.path.join(self.save_path, 'predict.npy'), pred_array)
        np.save(os.path.join(self.save_path, 'label.npy'), label_array)

        # Compute and dump metrics
        metrics = calculate_regression_metrics(y_true=label_array, y_pred=pred_array)
        metrics = convert_numpy(metrics)
        with open(
                os.path.join(self.save_path, 'metrics.json'), 'w', encoding='utf-8'
        ) as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
