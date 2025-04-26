# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 18:55
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 18:55
from typing import List, Any, Generator
import torch
from torch.autograd import Variable

"""
Module defining the CommonServer class for orchestrating federated learning
across multiple clients for electric vehicle data prediction.
"""


class CommonServer(object):
    """
    Server handler for federated learning aggregation and evaluation.

    This class manages training and evaluation clients, aggregates local model
    updates, and orchestrates global training and localization procedures.

    Attributes:
        train_clients (list): List of client instances for training.
        eval_clients (list): List of client instances for evaluation.
        aggregation (str): Aggregation strategy; e.g., 'fedavg'.
        model (object): Global model instance to be updated.
    """

    def __init__(
            self,
            train_clients: List[Any],
            eval_clients: List[Any],
            model: Any,
            aggregation: str = 'fedavg',
    ) -> None:
        """
        Initialize CommonServer with clients and aggregation settings.

        Args:
            train_clients (list): Clients participating in training.
            eval_clients (list): Clients reserved for evaluation.
            model (object): Global model instance to aggregate into.
            aggregation (str): Aggregation method; default is 'fedavg'.
        """
        super(CommonServer, self).__init__()
        self.train_clients = train_clients
        self.eval_clients = eval_clients
        self.aggregation = aggregation
        self.model = model

    def train(self, global_epochs: int, local_epochs: int) -> None:
        """
        Conduct federated training over multiple global rounds.

        Each global round performs local training on each train_client,
        aggregates their model updates, and then evaluates on eval_clients.

        Args:
            global_epochs (int): Number of global federated rounds.
            local_epochs (int): Number of local epochs per client.
        """
        for global_epoch in range(1, global_epochs + 1):
            local_models = []
            # Collect local model parameters from each training client
            for train_c in self.train_clients:
                train_c.refresh(self.model)
                train_c.train(now_epoch=global_epoch, local_epochs=local_epochs, save_model=False)
                local_models.append(train_c.get_model())

            # Aggregate local models into the global model
            self.aggregate(local_models)

            # Evaluate updated global model on evaluation clients
            for eval_c in self.eval_clients:
                eval_c.refresh(self.model)
                eval_c.test(now_epoch=global_epoch)

    def localize(self, now_epoch: int, deploy_epochs: int) -> None:
        """
        Perform model localization by iterative fine-tuning on eval_clients.

        Args:
            now_epoch (int): Current federated round or global epoch.
            deploy_epochs (int): Number of local deployment epochs to run.
        """
        for eval_c in self.eval_clients:
            eval_c.refresh(self.model)
            eval_c.localize(now_epoch=now_epoch, deploy_epochs=deploy_epochs)

    def aggregate(self, local_models: List[Generator[torch.Tensor, None, None]]) -> None:
        """
        Aggregate multiple local model parameter lists into the global model.

        Implements Federated Averaging (FedAvg) by summing each parameter
        weighted equally across clients.

        Args:
            local_models (list): List of parameter generators from clients.
        """
        num_clients = len(local_models)
        # Initialize global parameters to zero before averaging
        for idx, local_m in enumerate(local_models):
            for w_global, w_local in zip(self.model.model.parameters(), local_m):
                # Zero-init on first client iteration
                if idx == 0:
                    zero_tensor = Variable(torch.zeros_like(w_global))
                    w_global.data.copy_(zero_tensor.data)
                # Handle None local weights by zeroing
                if w_local is None:
                    w_local = Variable(torch.zeros_like(w_global))
                # Accumulate weighted average
                w_global.data.add_(w_local.data / num_clients)
