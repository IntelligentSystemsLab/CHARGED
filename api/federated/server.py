# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/13 18:55
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/13 18:55

from torch.autograd import Variable
import torch

class CommonServer(object):
    def __init__(
            self,
            train_clients,
            eval_clients,
            model,
            aggregation='fedavg',
    ):
        super(CommonServer, self).__init__()
        self.train_clients=train_clients
        self.eval_clients=eval_clients
        self.aggregation=aggregation
        self.model=model

    def train(self,global_epochs,local_epochs):
        for global_epoch in range(1,global_epochs+1):
            local_models=[]
            for train_c in self.train_clients:
                train_c.refresh(self.model)
                train_c.train(now_epoch=global_epoch,local_epochs=local_epochs,save_model=False)
                local_models.append(train_c.get_model())
            self.aggregate(local_models)
            for eval_c in self.eval_clients:
                eval_c.refresh(self.model)
                eval_c.test(now_epoch=global_epoch)

    def localize(self,now_epoch,deploy_epochs):
        for eval_c in self.eval_clients:
            eval_c.refresh(self.model)
            eval_c.localize(now_epoch=now_epoch,deploy_epochs=deploy_epochs)


    def aggregate(self,local_models):
        for local_m_id,local_m in enumerate(local_models):
            for w, w_t in zip(self.model.model.parameters(), local_m):
                if (w is None or local_m_id == 0):
                    w_tem = Variable(torch.zeros_like(w))
                    w.data.copy_(w_tem.data)
                if w_t is None:
                    w_t = Variable(torch.zeros_like(w))
                w.data.add_(w_t.data /len(local_models))
