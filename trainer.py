import time
import random
import uuid
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchinfo
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from normalization import MeanNormalize, MinMaxNormalize

from utils import set_gpu, make_directory, max_node_nums, num_graphs
from config import args


from model.backbone import Backbone
from model.REGPool import REGPool



class Trainer(object):
    def __init__(self, params):
        self.args = params

        # set GPU
        if self.args.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.args.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        # build the data
        self.args.use_node_attr = True
        self.data = None
        self.load_data()

        # build the model
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

    # load data
    def load_data(self):
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', self.args.dataset)
        print("Use node attr:", self.args.use_node_attr)
        print("Node attr norm:", self.args.data_normalization)
        if self.args.data_normalization == "MaxMin":
            data_normalization = MinMaxNormalize()
        elif self.args.data_normalization == "Mean":
            data_normalization = MeanNormalize()
        else:
            data_normalization = None
        dataset = TUDataset(path, self.args.dataset, use_node_attr=self.args.use_node_attr,
                            transform=data_normalization)
        dataset.data.edge_attr = None
        self.data = dataset
        self.data.max_node_nums = max_node_nums(self.data)
        print("Dataset:", self.data.data)
        print("Graph num:", self.data.len())
        print("Max node num:", self.data.max_node_nums)

    # load model
    def add_model(self):
        if self.args.model == '':
            model = None
        elif self.args.model == "Backbone":
            model = Backbone(self.data, self.args)
        elif self.args.model == "REGPool":
            model = REGPool(self.data, self.args)
        else:
            raise NotImplementedError
        model.to(self.device).reset_parameters()
        return model

    def add_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def add_lr_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_decay_step,
                                               gamma=self.args.lr_decay_factor)

    # save model locally
    def save_model(self, save_path):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'args': vars(self.args)
        }
        # print(save_path)
        torch.save(state, save_path)

    # load model from path
    def load_model(self, load_path):
        state = torch.load(load_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])

    # use 10-fold cross-validation
    def k_fold(self):
        kf = KFold(self.args.folds, shuffle=True, random_state=self.args.seed)

        test_indices, train_indices = [], []
        for _, idx in kf.split(torch.zeros(len(self.data)), self.data.data.y):
            test_indices.append(torch.from_numpy(idx))

        # test_indices = test_indices.type(torch.long)

        val_indices = [test_indices[i - 1] for i in range(self.args.folds)]
        for i in range(self.args.folds):
            train_mask = torch.ones(len(self.data), dtype=torch.uint8)
            ##
            test_indices[i] = test_indices[i].type(torch.long)
            val_indices[i] = val_indices[i].type(torch.long)
            ##
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero().view(-1))

        return train_indices, test_indices, val_indices

    # train model for an epoch
    def run_epoch(self, loader):
        self.model.train()

        loss_train = 0
        correct = 0
        extra_loss = 0
        cls_loss = 0
        d = 0
        for data in loader:
            self.optimizer.zero_grad()
            data = data.to(self.device)
            ground_truth = data.y.clone()
            out, items, d = self.model(data)
            # out = self.model(data)
            loss = F.nll_loss(out, ground_truth.view(-1))
            cls_loss += loss * num_graphs(data)

            # items.update({"CE loss": loss})
            # print(items)

            loss = loss + d
            # print("loss: ", loss)
            # loss = F.nll_loss(out, ground_truth.view(-1))
            loss.backward()
            loss_train += loss.item() * num_graphs(data)
            extra_loss += d * num_graphs(data)

            predict_out = out.max(dim=1)[1]
            correct += predict_out.eq(ground_truth).sum().item()
            self.optimizer.step()
            self.lr_scheduler.step()
        return loss_train / len(loader.dataset), cls_loss / len(loader.dataset), extra_loss / len(
            loader.dataset), correct / len(loader.dataset)

    # validate model
    def validate(self, loader):
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                out, _, _ = self.model(data)
                # out = self.model(data)

            loss = F.nll_loss(out, data.y, reduction='sum')  # 使用'sum'以便累计总损失
            pred = out.argmax(dim=1)
            correct = pred.eq(data.y.view(-1)).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += data.y.size(0)

        # 计算整个数据集的指标
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    # test model
    def predict(self, loader):
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        all_preds = []
        all_labels = []

        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                out, _, _ = self.model(data)
                # out = self.model(data)

            loss = F.nll_loss(out, data.y, reduction='sum')
            pred = out.argmax(dim=1)
            correct = pred.eq(data.y.view(-1)).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += data.y.size(0)

            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        cm = confusion_matrix(all_labels, all_preds)
        cr = classification_report(all_labels, all_preds, digits=4, output_dict=False)

        return avg_loss, accuracy, cm, cr

    # main function for running the experiments
    def run(self):
        val_accs, test_accs, train_times = [], [], []
        make_directory('{}/{}/'.format(self.args.directory_path, self.args.log_db))
        best_val_model_save_path = '{}/{}/seed_{}_best_val_model.pth'.format(self.args.directory_path, self.args.log_db,
                                                                             self.args.counter + 1)
        best_test_model_save_path = '{}/{}/seed_{}_best_test_model.pth'.format(self.args.directory_path,
                                                                               self.args.log_db, self.args.counter + 1)
        run_test_result_path = '{}/{}/seed_{}_test_result.txt'.format(self.args.directory_path, self.args.log_db,
                                                                      self.args.counter + 1)
        run_test_acc_path = '{}/{}/seed_{}_test_loss_and_acc.csv'.format(self.args.directory_path, self.args.log_db,
                                                                         self.args.counter + 1)
        run_test_acc_list = []
        best_fold_test_acc = 0

        if self.args.restore:
            self.load_model(best_val_model_save_path)
            print('Successfully Loaded previous model')

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # iterate over 10 folds
        for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*self.k_fold())):

            # Reinitialise model and optimizer for each fold
            self.model = self.add_model()
            self.optimizer = self.add_optimizer()
            self.lr_scheduler = self.add_lr_scheduler()

            train_dataset = self.data[train_idx]
            test_dataset = self.data[test_idx]
            val_dataset = self.data[val_idx]

            if 'adj' in train_dataset[0]:
                train_loader = DenseDataLoader(train_dataset, self.args.batch_size, shuffle=True)
                val_loader = DenseDataLoader(val_dataset, self.args.batch_size, shuffle=False)
                test_loader = DenseDataLoader(test_dataset, self.args.batch_size, shuffle=False)
            else:
                train_loader = DataLoader(train_dataset, self.args.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, self.args.batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, self.args.batch_size, shuffle=False)

            loss_save_path = '{}/{}/seed_{}_fold_{}_loss_and_acc.csv'.format(self.args.directory_path, self.args.log_db,
                                                                             self.args.counter + 1, fold + 1)
            loss_save_list = []

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            best_val_acc, best_test_acc = 0.0, 0.0
            best_val_loss = float('inf')
            best_epoch = 0
            patience_cnt = 0
            t = time.time()
            for epoch in range(1, self.args.max_epochs + 1):
                train_loss, cls_loss, extra_loss, train_acc = self.run_epoch(train_loader)
                val_loss, val_acc = self.validate(val_loader)

                print('Seed: {:02d}/{:02d}'.format(self.args.counter + 1, self.args.replication_num),
                      'Fold: {:02d}/{:02d}'.format(fold + 1, self.args.folds),
                      'Epoch: {:03d}'.format(epoch),
                      'loss_train: {:.6f}'.format(train_loss),
                      'cls_loss: {:.6f}'.format(cls_loss),
                      'extra_loss: {:.6f}'.format(extra_loss),
                      'acc_train: {:.6f}'.format(train_acc),
                      'loss_val: {:.6f}'.format(val_loss), 'acc_val: {:.6f}'.format(val_acc),
                      'time: {:.6f}s'.format(time.time() - t))

                loss_save_list.append({"Epoch": epoch, "loss_train": train_loss, "acc_train": train_acc,
                                       "loss_val": val_loss, "acc_val": val_acc})
                loss_save = pd.DataFrame(loss_save_list)
                loss_save.to_csv(loss_save_path, index=False)

                # save model for best val score
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(best_val_model_save_path)
                    best_epoch = epoch
                    patience_cnt = 0
                elif val_acc == best_val_acc and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(best_val_model_save_path)
                    best_epoch = epoch
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if patience_cnt == self.args.patience:
                    break

            train_times.append(time.time() - t)
            print('Fold {} Optimization Finished! Total time elapsed: {:.6f} s Best epoch: {:02d}'.format(fold + 1,
                                                                                                          time.time() - t,
                                                                                                          best_epoch))
            # load best model for testing
            self.load_model(best_val_model_save_path)
            best_test_loss, best_test_acc, cm, cr = self.predict(test_loader)
            print('Seed: {:02d}/{:02d}'.format(self.args.counter + 1, self.args.replication_num),
                  'Fold: {:02d}/{:02d}'.format(fold + 1, self.args.folds),
                  'loss_test: {:.6f}'.format(best_test_loss), 'acc_test: {:.6f}'.format(best_test_acc)
                  )
            print("Confusion Matrix: \n", cm)
            print("Confusion Report: \n", cr)

            run_test_acc_list.append({"Fold": fold + 1, 'loss_test': best_test_loss, 'acc_test': best_test_acc})
            run_test_acc_save = pd.DataFrame(run_test_acc_list)
            run_test_acc_save.to_csv(run_test_acc_path, index=False)

            with open(run_test_result_path, mode="a") as f:
                f.write(
                    'Fold {} Optimization Finished! Total time elapsed: {:.6f} s Best epoch: {:02d} \n'.format(fold + 1,
                                                                                                               time.time() - t,
                                                                                                               best_epoch))
                f.write('loss_test: {:.6f},  acc_test: {:.6f} \n'.format(best_test_loss, best_test_acc))
                f.write("Confusion Matrix: \n")
                print(cm, file=f)
                f.write("Confusion Report: \n")
                print(cr, file=f)

            if best_test_acc > best_fold_test_acc:
                best_fold_test_acc = best_test_acc
                self.save_model(best_test_model_save_path)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            val_accs.append(best_val_acc)
            test_accs.append(best_test_acc)

        train_time_mean = np.round(np.mean(train_times), 6)
        val_acc_mean = np.round(np.mean(val_accs), 6)
        test_acc_mean = np.round(np.mean(test_accs), 6)

        train_time_std = np.round(np.std(train_times), 6)
        val_acc_std = np.round(np.std(val_accs), 6)
        test_acc_std = np.round(np.std(test_accs), 6)

        return val_acc_mean, val_acc_std, test_acc_mean, test_acc_std, train_time_mean, train_time_std


def train():
    if not args.restore:
        args.name = time.strftime('%Y_%m_%d') + '_' + time.strftime('%H_%M_%S') + '_' + args.name
    # Model training
    print('Training Start ...')

    if args.replication:
        # Random seeds for replication
        seeds = [3653]
    else:
        seeds = [args.seed]
    args.replication_num = len(seeds)

    counter = 0
    args.log_db = args.name
    print("log_db:", args.log_db)
    args.directory_path = 'results/{}/{}/layer{}_batch{}_hid{}_epoch{}_early{}_pooling{}_drop{}_lr{}_decay{}_JK_{}_others{}/'.format(
        args.model, args.dataset, args.num_layers, args.batch_size, args.hid_dim, args.max_epochs, args.patience,
        args.pooling_ratio, args.dropout_ratio, args.lr, args.weight_decay, args.jump_connection, args.notes)
    print(args)
    make_directory(args.directory_path)
    test_result_path = '{}/{}_all_results.txt'.format(args.directory_path, args.log_db)
    test_acc_path = '{}/{}_all_loss_and_acc.csv'.format(args.directory_path, args.log_db)
    test_acc_list = []

    avg_val = []
    avg_test = []
    avg_time = []
    for seed in seeds:
        # Set seed
        args.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)  # 固定当前GPU
            torch.cuda.manual_seed_all(args.seed)  # 固定所有GPU

        set_gpu(args.gpu)
        args.counter = counter
        args.name = '{}_seed_{}'.format(args.log_db, counter)

        with open(test_result_path, mode="a") as f:
            print(args, file=f)

        # start training the model
        model = Trainer(args)
        val_acc, val_acc_std, test_acc, test_acc_std, train_time, train_time_std = model.run()
        print('For seed {}  Val Accuracy Mean: {:.6f} ± {:.6f}  Test Accuracy Mean: {:.6f} ± {:.6f}  Train Time Mean: '
              '{:.6f} ± {:.6f} \n'.format(seed, val_acc,
                                          val_acc_std,
                                          test_acc,
                                          test_acc_std,
                                          train_time,
                                          train_time_std))
        with open(test_result_path, mode="a") as f:
            f.write(
                'For seed {} ({:02d}/{:02d})  Val Accuracy: {:.6f}   Test Accuracy: {:.6f}   Train Time: {:.6f} s\n'.format(
                    seed, counter + 1, args.replication_num, val_acc, test_acc, train_time))

        if counter == 0:
            with open(test_result_path, mode="a") as f:
                print(torchinfo.summary(model=model.model), file=f)

        test_acc_list.append(
            {"Seed No.": counter + 1, "Seed": seed, "Val Accuracy": val_acc, "Test Accuracy": test_acc})
        test_acc_save = pd.DataFrame(test_acc_list)
        test_acc_save.to_csv(test_acc_path, index=False)

        avg_val.append(val_acc)
        avg_test.append(test_acc)
        avg_time.append(train_time)
        counter += 1

    print('Val Accuracy: {:.4f} ± {:.4f} Test Accuracy: {:.4f} ± {:.4f} Train Time: {:.3f} ± {:.3f} s'.format(
        np.mean(avg_val), np.std(avg_val), np.mean(avg_test), np.std(avg_test), np.mean(avg_time), np.std(avg_time)))
    with open(test_result_path, mode="a") as f:
        f.write('Val Accuracy: {:.4f} ± {:.4f} Test Accuracy: {:.4f} ± {:.4f} Train Time: {:.4f} ± {:.4f} s'.format(
            np.mean(avg_val), np.std(avg_val), np.mean(avg_test), np.std(avg_test), np.mean(avg_time),
            np.std(avg_time)))
    return args.log_db, np.mean(avg_val), np.std(avg_val), np.mean(avg_test), np.std(avg_test), np.mean(
        avg_time), np.std(avg_time)


if __name__ == '__main__':
    train()
