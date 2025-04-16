# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 19:19
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 19:19


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from api.parsing.federated import federated_parse_args
from api.model.config import PredictionModel
from api.trainer.common import PredictionTrainer
from api.utils import random_seed, get_n_feature, Logger, get_data_paths
from api.dataset.distributed import DistributedEVDataset
from api.federated.client import CommonClient
from api.federated.server import CommonServer
from api.trainer.federated import ClientTrainer

if __name__ == '__main__':
    args = federated_parse_args()
    if args.pred_type == 'station':
        base_path = f'{args.output_path}{args.pred_type}/{args.city}/{args.model}-{args.feature}-{args.auxiliary}-{args.seq_l}-{args.pre_len}-{args.max_stations}-{args.eval_percentage}'
    else:
        base_path = f'{args.output_path}{args.pred_type}/{args.city}/{args.model}-{args.feature}-{args.auxiliary}-{args.seq_l}-{args.pre_len}-{args.max_stations}-{args.eval_city}'
    new_path = base_path
    counter = 0
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}#{counter}/"
    os.makedirs(new_path)
    sys.stdout = Logger(os.path.join(new_path, 'logging.txt'))

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    random_seed(seed=args.seed)
    data_paths = get_data_paths(ori_path=args.data_path, cities=args.city, suffix='_remove_zero')

    ev_dataset = DistributedEVDataset(
        feature=args.feature,
        auxiliary=args.auxiliary,
        data_paths=data_paths,
        pred_type=args.pred_type,
        eval_percentage=args.eval_percentage,
        eval_city=args.eval_city,
        max_stations=args.max_stations,
    )
    print(
        f"Running cross stations evaluation on {args.city} with feature={args.feature}, pre_l={args.pre_len}, model={args.model}, auxiliary={args.auxiliary}, pred_type(node)={args.pred_type}")
    train_clients = []
    train_clients_id = []
    eval_clients = []
    eval_clients_id = []
    for client_id, data_dict in ev_dataset.training_clients_data.items():
        client_path = f'{new_path}{client_id}/'
        # os.makedirs(client_path)
        train_clients.append(
            CommonClient(
                client_id=client_id,
                data_dict=data_dict,
                scaler=ev_dataset.city_scalers[client_id[:3]],
                model_module=PredictionModel,
                trainer_module=ClientTrainer,
                seq_l=args.seq_l,
                pre_len=args.pre_len,
                model_name=args.model,
                n_fea=ev_dataset.n_fea,
                batch_size=args.batch_size,
                device=args.device,
                save_path=client_path,
                support_rate=1,
            )
        )
        train_clients_id.append(client_id)
    print('Training on:')
    print(train_clients_id)
    for client_id, data_dict in ev_dataset.eval_clients_data.items():
        client_path = f'{new_path}{client_id}/'
        os.makedirs(client_path)
        eval_clients.append(
            CommonClient(
                client_id=client_id,
                data_dict=data_dict,
                scaler=ev_dataset.city_scalers[client_id[:3]],
                model_module=PredictionModel,
                trainer_module=ClientTrainer,
                seq_l=args.seq_l,
                pre_len=args.pre_len,
                model_name=args.model,
                n_fea=ev_dataset.n_fea,
                batch_size=args.batch_size,
                device=args.device,
                save_path=client_path,
                support_rate=0.5,
            )
        )
        eval_clients_id.append(client_id)
    print('Evaluation on:')
    print(eval_clients_id)
    ev_server = CommonServer(
        train_clients=train_clients,
        eval_clients=eval_clients,
        model=PredictionModel(
            num_node=1,
            n_fea=ev_dataset.n_fea,
            model_name=args.model,
            seq_l=args.seq_l,
            pre_len=args.pre_len,
        ),
    )
    ev_server.train(global_epochs=args.global_epoch,local_epochs=args.local_epoch)
    ev_server.localize(now_epoch=args.global_epoch,deploy_epochs=args.deploy_epoch)
