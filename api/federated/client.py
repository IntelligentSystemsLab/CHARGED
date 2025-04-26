# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 18:55
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 18:55

from typing import Dict, Any, Callable, Optional, Generator
import torch
from torch.utils.data import DataLoader, Subset

from api.utils import CreateDataset

"""
Module defining the CommonClient class for local client-side training and evaluation
in federated learning for electric vehicle data prediction.
"""


class CommonClient(object):
    """
    Client handler for federated learning on electric vehicle datasets.

    This class initializes client-specific datasets, model, and trainer,
    and provides methods for local training, testing, model localization,
    parameter refresh, and parameter retrieval.

    Attributes:
        client_id (str): Unique identifier for the client.
        scaler (object): Scaler used to inverse-transform predictions.
        feat (np.ndarray): Feature array for this client [time, nodes].
        extra_feat (np.ndarray or None): Auxiliary features array.
        trainer (object): Trainer instance handling training and testing logic.
    """

    def __init__(
            self,
            client_id: str,
            data_dict: Dict[str, Any],
            scaler: Any,
            model_module: Callable[..., Any],
            trainer_module: Callable[..., Any],
            seq_l: int,
            pre_len: int,
            model_name: str,
            n_fea: int,
            batch_size: int,
            device: torch.device,
            save_path: str,
            support_rate: float = 1.0,
    ) -> None:
        """
        Initialize CommonClient by constructing dataset subsets,
        model, and trainer.

        Args:
            client_id (str): Identifier for the client.
            data_dict (dict): Contains 'feat', 'extra_feat', and 'time'.
            scaler (object): Scaler fitted on training data.
            model_module (callable): Class or factory for model instantiation.
            trainer_module (callable): Class or factory for trainer instantiation.
            seq_l (int): Input sequence length.
            pre_len (int): Prediction horizon length.
            model_name (str): Name of the model architecture.
            n_fea (int): Number of auxiliary feature channels.
            batch_size (int): Batch size for DataLoaders.
            device (torch.device): Torch device for computations.
            save_path (str): Path to save trained model parameters.
            support_rate (float): Fraction of local data for training.
        """
        super(CommonClient, self).__init__()

        # Client properties
        self.client_id = client_id
        self.scaler = scaler
        self.feat = data_dict['feat']
        self.extra_feat = data_dict['extra_feat']

        # Initialize model
        model = model_module(
            num_node=1,
            n_fea=n_fea,
            model_name=model_name,
            seq_l=seq_l,
            pre_len=pre_len,
        )

        # Create full dataset and split into support and test sets
        dataset = CreateDataset(seq_l, pre_len, self.feat, self.extra_feat, device)
        total_samples = len(dataset)
        support_count = int(total_samples * support_rate)
        train_indices = list(range(support_count))
        test_indices = list(range(support_count, total_samples))

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        # DataLoaders for local training and testing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Determine if auxiliary features are present
        if self.extra_feat is None:
            extra_feat_tag = False
        else:
            extra_feat_tag = True

        # Initialize trainer with data loaders and model
        self.trainer = trainer_module(
            train_loader=train_loader,
            test_loader=test_loader,
            extra_feat_tag=extra_feat_tag,
            model=model,
            save_path=save_path,
            scaler=scaler,
            device=device,
        )

    def train(
            self,
            now_epoch: int,
            local_epochs: int,
            save_model: bool = False,
    ) -> None:
        """
        Perform local training for a specified number of epochs.

        Args:
            now_epoch (int): Current federated round or global epoch.
            local_epochs (int): Number of local epochs to train.
            save_model (bool): Whether to save model weights after training.
        """
        self.trainer.now_epoch = now_epoch
        self.trainer.deploy_epoch = 1
        self.trainer.training(epoch=local_epochs, save_model=save_model)

    def test(
            self,
            now_epoch: int,
            model_path: Optional[str] = None,
    ) -> None:
        """
        Evaluate the local model on the client's test dataset.

        Args:
            now_epoch (int): Current federated or local epoch.
            model_path (str, optional): Path to a saved model for testing.
        """
        self.trainer.now_epoch = now_epoch
        self.trainer.test(model_path=model_path)

    def localize(
            self,
            now_epoch: int,
            deploy_epochs: int,
            save_model: bool = False,
            model_path: Optional[str] = None,
    ) -> None:
        """
        Perform iterative local training and testing for model localization.

        Args:
            now_epoch (int): Current federated round or global epoch.
            deploy_epochs (int): Number of deployment epochs to run.
            save_model (bool): Whether to save model weights after each epoch.
            model_path (str, optional): Path to a saved model for testing.
        """
        self.trainer.now_epoch = now_epoch
        for deploy_epoch in range(1, deploy_epochs + 1):
            self.trainer.deploy_epoch = deploy_epoch
            self.trainer.training(epoch=1, save_model=save_model)
            self.trainer.test(model_path=model_path)

    def refresh(self, model: Any) -> None:
        """
        Update the local model parameters from a global model.

        Args:
            model (object): Global model instance with .model.parameters().
        """
        for w_local, w_global in zip(
                self.trainer.ev_model.model.parameters(),
                model.model.parameters()
        ):
            w_local.data.copy_(w_global.data)

    def get_model(self) -> Generator[torch.Tensor, None, None]:
        """
        Retrieve the local model's current parameters.

        Returns:
            generator: Iterable of torch.Tensor parameters.
        """
        return self.trainer.ev_model.model.parameters()
