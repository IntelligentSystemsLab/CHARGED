# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 18:55
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 18:55
from torch.utils.data import DataLoader, Subset

from api.trainer.federated import ClientTrainer
from api.utils import CreateDataset


class CommonClient(object):
    def __init__(
            self,
            client_id,
            data_dict,
            scaler,
            model_module,
            trainer_module,
            seq_l,
            pre_len,
            model_name,
            n_fea,
            batch_size,
            device,
            save_path,
            support_rate=1,
    ):
        super(CommonClient, self).__init__()
        self.client_id=client_id
        self.scaler=scaler
        self.feat=data_dict['feat']
        self.extra_feat=data_dict['extra_feat']
        model=model_module(
            num_node=1,
            n_fea=n_fea,
            model_name=model_name,
            seq_l=seq_l,
            pre_len=pre_len,
        )
        dataset=CreateDataset(seq_l,pre_len,self.feat, self.extra_feat, device)
        total_samples = len(dataset)
        support_count = int(total_samples * support_rate)
        train_indices = list(range(support_count))
        test_indices = list(range(support_count, total_samples))

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        if self.extra_feat is None:
            extra_feat_tag=False
        else:
            extra_feat_tag=True
        self.trainer = trainer_module(
            train_loader=train_loader,
            test_loader=test_loader,
            extra_feat_tag=extra_feat_tag,
            model=model,
            save_path=save_path,
            scaler=scaler,
            device=device,
        )

    def train(self,now_epoch,local_epochs,save_model=False):
        self.trainer.now_epoch=now_epoch
        self.trainer.deploy_epoch=1
        self.trainer.training(epoch=local_epochs,save_model=save_model)

    def test(self,now_epoch,model_path=None):
        self.trainer.now_epoch=now_epoch
        self.trainer.test(model_path=model_path)

    def localize(self,now_epoch,deploy_epochs,save_model=False,model_path=None):
        self.trainer.now_epoch = now_epoch
        for deploy_epoch in range(1,deploy_epochs+1):
            self.trainer.deploy_epoch =deploy_epoch
            self.trainer.training(epoch=1, save_model=save_model)
            self.trainer.test(model_path=model_path)

    def refresh(self, model):
        for w, w_t in zip(self.trainer.ev_model.model.parameters(), model.model.parameters()):
            w.data.copy_(w_t.data)

    def get_model(self):
        return self.trainer.ev_model.model.parameters()