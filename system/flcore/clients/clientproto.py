import copy
import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict


class clientProto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda


    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)
        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model.eval()

        if global_protos is None:
            return super().test_metrics()

        total_correct = 0
        total_samples = 0
        y_prob = []
        y_true = []
        preds_all = []
        targets_all = []
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)

                output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                for i, r in enumerate(rep):
                    for j, pro in global_protos.items():
                        if type(pro) != type([]):
                            output[i, j] = self.loss_mse(r, pro)

                preds = torch.argmin(output, dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.shape[0]
                if self.use_bacc_metric:
                    preds_all.append(preds.detach().cpu().numpy())
                    targets_all.append(y.detach().cpu().numpy())

                scores = (-output).detach().cpu().numpy()
                y_prob.append(scores)
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        if total_samples == 0:
            return (float("nan") if self.use_bacc_metric else 0.0), 0, float("nan")

        y_prob = np.concatenate(y_prob, axis=0) if y_prob else np.array([])
        y_true = np.concatenate(y_true, axis=0) if y_true else np.array([])
        try:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        except ValueError:
            auc = float("nan")

        if self.use_bacc_metric:
            if len(preds_all) == 0:
                return float("nan"), total_samples, auc
            preds_cat = np.concatenate(preds_all, axis=0)
            targets_cat = np.concatenate(targets_all, axis=0)
            bacc = metrics.balanced_accuracy_score(targets_cat, preds_cat)
            return bacc, total_samples, auc

        return total_correct, total_samples, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos
