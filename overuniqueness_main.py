import torch
from models.dln import DecoupleModel, iGNN, iGNN_V2, iGNN_energy_version, iGNN_energy_version_fc
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import json
import matplotlib.pyplot as plt
# import math
import random
import numpy as np
# from torchinfo import summary
from torch_geometric.data import Data
from utils.wl_test import wl_relabel, wl_train_test_ood
import argparse
from datetime import datetime
from pathlib import Path
import sys
from utils.text_hyperparameters import add_hyperparameter_text
from utils.dataset import load_dataset, load_large_dataset
from utils.data_split_util import rand_train_test_idx
from utils.timestamp import get_timestamp
from torch_geometric.utils import to_undirected
from utils.over_smoothing_measure import dirichlet_energy
import gc
from torch_geometric.utils import add_self_loops

# TODO: improve result folder structure
# TODO: dashed line for std result after running multiple runs


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tensor_memory_mb(tensor):
    return tensor.element_size() * tensor.nelement() / 1024 ** 2

def get_pyg_data_gpu_memory_mb(data):
    total_mem = 0
    for key, val in data.items():
        if torch.is_tensor(val) and val.is_cuda:
            mem = val.element_size() * val.nelement() / 1024 ** 2
            print(f'{key}: {val.shape} -> {mem:.2f} MB')
            total_mem += mem
    print(f'Total: {total_mem:.2f} MB')
    return total_mem

def train(model, data, train_idx, optimizer, criterion, energy_lambda, energy_threshold):
    model.train()
    out = model(data)
    # out, embedding, norms_per_layer = model(data)
    # energy_loss = dirichlet_energy(embedding, data.edge_index)
    loss = criterion(out[train_idx], data.y[train_idx])
    train_learning_loss = loss
    # if energy_loss > energy_threshold:
    #     loss += energy_loss * energy_lambda
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # return loss.item(), energy_loss.item(), train_learning_loss.item(), norms_per_layer
    return loss.item()

@torch.no_grad()
def evaluate(model, criterion, data, train_split_idx, validation_split_idx, test_split_idx, energy_lambda):
    model.eval()
    # out, embedding, _ = model(data)
    out = model(data)
    # energy_loss = dirichlet_energy(embedding, data.edge_index)
    valid_learning_loss = criterion(out[validation_split_idx], data.y[validation_split_idx])
    # valid_loss = valid_learning_loss + energy_loss * energy_lambda
    test_learning_loss = criterion(out[test_split_idx], data.y[test_split_idx])
    # test_loss = test_learning_loss + energy_loss * energy_lambda

    y_true_train = data.y[train_split_idx]
    y_pred_train = out[train_split_idx]
    y_pred_train = y_pred_train.argmax(dim=-1)
    y_true_validation = data.y[validation_split_idx]
    y_pred_validation = out[validation_split_idx]
    y_pred_validation = y_pred_validation.argmax(dim=-1)
    y_true_test = data.y[test_split_idx]
    y_pred_test = out[test_split_idx]
    y_pred_test = y_pred_test.argmax(dim=-1)

    correct_train = y_true_train == y_pred_train
    correct_valid = y_true_validation == y_pred_validation
    correct_tes = y_true_test == y_pred_test

    train_acc = correct_train.sum().item() / len(correct_train)
    valid_acc = correct_valid.sum().item() / len(correct_valid)
    test_acc = correct_tes.sum().item() / len(correct_tes)

    # return train_acc, valid_acc, test_acc, valid_loss.item(), test_loss.item(), energy_loss.item(), \
    #         valid_learning_loss.item(), test_learning_loss.item()

    return train_acc, valid_acc, test_acc, \
            valid_learning_loss.item(), test_learning_loss.item()

### Training loop ###
# for run in range(args.runs):
#     # split_idx = split_idx_lst[run]
#     # train_mask = torch.zeros(n, dtype=torch.bool)
#     # train_mask[split_idx['train']] = True

#     # model.reset_parameters()
#     # optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
#     # best_val = float('-inf')
#     # best_test = float('-inf')
#     # if args.save_model:
#     #     save_model(args, model, optimizer, run)
#     num_batch = n // args.batch_size + 1

#     for epoch in range(args.epochs):

#         model.to(device)
#         model.train()

#         loss_train = 0
#         idx = torch.randperm(n)
#         for i in range(num_batch):
#             idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
#             train_mask_i = train_mask[idx_i]
#             x_i = x[idx_i].to(device)
#             edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
#             edge_index_i = edge_index_i.to(device)
#             y_i = true_label[idx_i].to(device)
#             optimizer.zero_grad()
#             out_i = model(x_i, edge_index_i)
#             out_i = F.log_softmax(out_i, dim=1)
#             loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i])
#             loss.backward()
#             optimizer.step()
#             loss_train += loss.item()
#         loss_train /= num_batch

#         if epoch % args.eval_step == 0 and epoch > args.eval_epoch:
#             result = evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, device)


# def run_large(dataset_name, num_mp_layers, mp_hidden_dim,
#         optimizer_lr, loss_func, total_epoch, freeze, save_model=False, skip_connection=False, dropout=0,
#         folder_name_suffix="", energy_lambda=1.0, energy_threshold=0, batch_size=10000):
#     fix_seed()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     display_step = 20
#     data, train_idx, valid_idx, test_idx = load_large_dataset(data_dir='data', dataset_name=dataset_name)
#     data.edge_index = to_undirected(data.edge_index)
#     d = data.x.shape[1]
#     c = data.y.max().item() + 1
#     n = data.x.shape[0]

#     train_mask = torch.zeros(n, dtype=torch.bool)
#     train_mask[train_idx] = True

#     model = iGNN (
#         in_dim=d,
#         out_dim=c,
#         mp_width=mp_hidden_dim,
#         fl_width=fl_hidden_dim,
#         num_mp_layers = num_mp_layers,
#         num_fl_layers = num_fl_layers,
#         freeze=freeze,
#         skip_connection=skip_connection,
#         dropout=0
#     ).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_lr)
#     criterion = torch.nn.CrossEntropyLoss()
#     if loss_func == 'CrossEntropyLoss':
#         criterion = torch.nn.CrossEntropyLoss()
#     elif loss_func == 'NLLLoss':
#         criterion = torch.nn.NLLLoss()

#     print('Experiment run')
#     print(f'dataset: {dataset_name}')
#     print(f'num_mp_layers: {num_mp_layers}')
#     print(f'mp_hidden_dim: {mp_hidden_dim}')
#     print(f'optimizer_lr: {optimizer_lr}')
#     print(f'loss_func: {loss_func}')
#     print(f'total_epoch: {total_epoch}')
#     print(f'energy_lambda: {energy_lambda}')


#     train_loss_list = []
#     valid_loss_list = []
#     test_loss_list = []
#     train_learning_loss_list = []
#     valid_learning_loss_list = []
#     test_learning_loss_list = []
#     best_val = float('-inf')
#     best_test = float('-inf')
#     train_accuracy_list = []
#     valid_accuracy_list = []
#     test_accuracy_list = []
#     energy_loss_list = []
#     norms_list = []

#     num_batch = n // batch_size + 1

#     for epoch in tqdm(range(1,total_epoch)):
#         loss, energy_loss, train_learning_loss, norms_per_layer = train(model,data,train_idx,optimizer,criterion, energy_lambda, energy_threshold)
#         train_acc, valid_acc, test_acc, valid_loss, test_loss, _, valid_learning_loss, test_learning_loss  = evaluate(model, criterion, data, train_idx, valid_idx, test_idx, energy_lambda)

#         train_accuracy_list.append(train_acc)
#         valid_accuracy_list.append(valid_acc)
#         test_accuracy_list.append(test_acc)
#         train_loss_list.append(loss)
#         valid_loss_list.append(valid_loss)
#         test_loss_list.append(test_loss)
#         energy_loss_list.append(energy_loss)
#         train_learning_loss_list.append(train_learning_loss)
#         valid_learning_loss_list.append(valid_learning_loss)
#         test_learning_loss_list.append(test_learning_loss)
#         norms_list.append(norms_per_layer)

#         if test_acc > best_test:
#             best_test = test_acc
#         if valid_acc > best_val:
#             best_val = valid_acc

#         if epoch % display_step == 0:
#             print(f'Epoch: {epoch:02d}, '
#                 f'Loss: {loss:.4f}, '
#                 f'Energy: {energy_loss:.4f}, '
#                 f'Train: {100 * train_acc:.2f}%, '
#                 f'Valid: {100 * valid_acc:.2f}%, '
#                 f'Test: {100 * test_acc:.2f}%, '
#                 f'Best Valid: {100 * best_val:.2f}%, '
#                 f'Best Test: {100 * best_test:.2f}%')

#     print(f'train_accuracy_list: {train_accuracy_list}')
#     print(f'valid_accuracy_list: {valid_accuracy_list}')
#     print(f'test_accuracy_list: {test_accuracy_list}')
#     print(f'best validation: {best_val}')
#     print(f'best test: {best_test}')

#     # Plotting the loss, min cell is 0.1, large figure
#     params = {
#         'dataset_name': dataset_name,
#         'num_mp_layers': num_mp_layers,
#         'mp_hidden_dim': mp_hidden_dim,
#         'optimizer_lr': optimizer_lr,
#         'freeze': freeze,
#         # 'skip_connection': skip_connection,
#         # 'dropout': dropout
#         'energy lambda': energy_lambda
#     }
    
#     # in case run() is executed in other files other than main.py
#     try:
#         timestamp
#     except NameError:
#         timestamp = get_timestamp()
#     try:
#         folder_name
#     except NameError:
#         # Create folder for results
#         folder = Path(f"result_energy_loss")
#         folder.mkdir(parents=True, exist_ok=True)
#         folder_name = folder.name
#         # folder = Path(f"result_{dataset_name}_{folder_name_suffix}")
#         # folder.mkdir(parents=True, exist_ok=True)
#         # folder_name = folder.name

#     fig, ax = add_hyperparameter_text(params)
#     # plt.figure(figsize=(10, 5))
#     ax.plot(train_loss_list, label='Train Loss', color='blue')
#     ax.plot(valid_loss_list, label='Valid Loss', color='orange')
#     ax.plot(test_loss_list, label='Test Loss', color='green')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Loss vs Epochs')
#     plt.legend()
#     plt.savefig('{}/loss_{}_{}.png'.format(folder_name, dataset_name, timestamp))
#     # plt.clf()  # Clear the current figure for the next plot
#     plt.close()
#     # Plotting the acc in one figure
#     fig, ax = add_hyperparameter_text(params)
#     # plt.figure(figsize=(10, 5))
#     ax.plot(train_accuracy_list, label='Train Accuracy', color='blue')
#     ax.plot(valid_accuracy_list, label='Valid Accuracy', color='orange')
#     ax.plot(test_accuracy_list, label='Test Accuracy', color='green')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy vs Epochs')
#     plt.legend()
#     plt.savefig('{}/accuracy_{}_{}.png'.format(folder_name, dataset_name, timestamp))
#     # plt.clf()  # Clear the current figure for the next plot
#     plt.close()
#     # Plotting the energy in one figure
#     fig, ax = add_hyperparameter_text(params)
#     # plt.figure(figsize=(10, 5))
#     ax.plot(energy_loss_list, label='Dirichlet Energy', color='blue')
#     plt.xlabel('Epochs')
#     plt.ylabel('Energy')
#     plt.title('Energy vs Epochs')
#     plt.legend()
#     plt.savefig('{}/energy_{}_{}.png'.format(folder_name, dataset_name, timestamp))
#     # plt.clf()  # Clear the current figure for the next plot
#     plt.close()

#     # if save_model:
#     #     torch.save(model.state_dict(), F'saved_models/model_weights_{dataset_name}_{num_mp_layers}_{mp_hidden_dim}_{num_fl_layers}_{fl_hidden_dim}_{dropout}.pth')

#     return best_val, best_test, model, train_accuracy_list, valid_accuracy_list, test_accuracy_list, energy_loss_list, \
#             train_loss_list, valid_loss_list, test_loss_list, train_learning_loss_list, \
#             valid_learning_loss_list, test_learning_loss_list, norms_list




def run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim, fl_hidden_dim,
        optimizer_lr, loss_func, total_epoch, freeze, save_model=False, skip_connection=False, dropout=0,
        folder_name_suffix="", energy_lambda=1.0, energy_threshold=10000):
    fix_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    display_step = 5
    if not dataset_name.startswith("ogbn"):
        data = load_dataset(data_dir='data', dataset_name=dataset_name)
    else:
        data, train_idx, valid_idx, test_idx = load_large_dataset(data_dir='data', name=dataset_name)
        data.y = data.y.squeeze(-1)
    data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.shape[0])
    print(data)
    data.to(device)
    d = data.x.shape[1]
    c = data.y.max().item() + 1

    # data split for train, val, and test
    if hasattr(data, 'train_mask'):
        train_idx = torch.where(data.train_mask)[0]
        valid_idx = torch.where(data.val_mask)[0]
        test_idx = torch.where(data.test_mask)[0]

        unlabeled_mask = ~(data.train_mask | data.val_mask | data.test_mask)
        unlabeled_idx = torch.where(unlabeled_mask)[0]
        # num_samples = int(len(unlabeled_idx) * extra_train_data_rate)

        # Randomly shuffle and select a portion
        # selected_unlabeled_idx = unlabeled_idx[torch.randperm(len(unlabeled_idx))[:num_samples]]
        # train_idx = torch.cat([train_idx, unlabeled_idx])
    else:
        if not dataset_name.startswith("ogbn"):
            train_idx, valid_idx, test_idx = rand_train_test_idx(data.y, train_prop=0.6, valid_prop=0.2)

    model = iGNN (
        in_dim=d,
        out_dim=c,
        mp_width=mp_hidden_dim,
        fl_width=fl_hidden_dim,
        num_mp_layers = num_mp_layers,
        num_fl_layers = num_fl_layers,
        freeze=freeze,
        skip_connection=skip_connection,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_lr)
    criterion = torch.nn.CrossEntropyLoss()
    if loss_func == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_func == 'NLLLoss':
        criterion = torch.nn.NLLLoss()

    print('Experiment run')
    print(f'dataset: {dataset_name}')
    print(f'num_mp_layers: {num_mp_layers}')
    print(f'mp_hidden_dim: {mp_hidden_dim}')
    print(f'optimizer_lr: {optimizer_lr}')
    print(f'loss_func: {loss_func}')
    print(f'total_epoch: {total_epoch}')
    print(f'energy_lambda: {energy_lambda}')


    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []
    train_learning_loss_list = []
    valid_learning_loss_list = []
    test_learning_loss_list = []
    best_val = float('-inf')
    best_test = float('-inf')
    train_accuracy_list = []
    valid_accuracy_list = []
    test_accuracy_list = []
    energy_loss_list = []
    norms_list = []

    for epoch in tqdm(range(1,total_epoch)):
        # loss, energy_loss, train_learning_loss, norms_per_layer = train(model,data,train_idx,optimizer,criterion, energy_lambda, energy_threshold)
        # train_acc, valid_acc, test_acc, valid_loss, test_loss, _, valid_learning_loss, test_learning_loss  = evaluate(model, criterion, data, train_idx, valid_idx, test_idx, energy_lambda)

        loss = train(model,data,train_idx,optimizer,criterion, energy_lambda, energy_threshold)
        train_acc, valid_acc, test_acc, valid_learning_loss, test_learning_loss  = evaluate(model, criterion, data, train_idx, valid_idx, test_idx, energy_lambda)

        train_accuracy_list.append(train_acc)
        valid_accuracy_list.append(valid_acc)
        test_accuracy_list.append(test_acc)
        train_loss_list.append(loss)
        # valid_loss_list.append(valid_loss)
        # test_loss_list.append(test_loss)
        # energy_loss_list.append(energy_loss)
        # train_learning_loss_list.append(train_learning_loss)
        valid_learning_loss_list.append(valid_learning_loss)
        test_learning_loss_list.append(test_learning_loss)
        # norms_list.append(norms_per_layer)

        if test_acc > best_test:
            best_test = test_acc
        if valid_acc > best_val:
            best_val = valid_acc

        if epoch % display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                # f'Energy: {energy_loss:.4f}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}%, '
                f'Test: {100 * test_acc:.2f}%, '
                f'Best Valid: {100 * best_val:.2f}%, '
                f'Best Test: {100 * best_test:.2f}%')

    print(f'train_accuracy_list: {train_accuracy_list}')
    print(f'valid_accuracy_list: {valid_accuracy_list}')
    print(f'test_accuracy_list: {test_accuracy_list}')
    print(f'best validation: {best_val}')
    print(f'best test: {best_test}')

    # Plotting the loss, min cell is 0.1, large figure
    params = {
        'dataset_name': dataset_name,
        'num_mp_layers': num_mp_layers,
        'mp_hidden_dim': mp_hidden_dim,
        'num_fl_layers': num_fl_layers,
        'fl_hidden_dim': fl_hidden_dim,
        'optimizer_lr': optimizer_lr,
        'freeze': freeze,
        'skip_connection': skip_connection,
        'dropout': dropout
        # 'energy lambda': energy_lambda
    }
    
    # in case run() is executed in other files other than main.py
    try:
        timestamp
    except NameError:
        timestamp = get_timestamp()
    try:
        folder_name
    except NameError:
        # Create folder for results
        folder = Path(f"result_overuniqueness_loss")
        folder.mkdir(parents=True, exist_ok=True)
        folder_name = folder.name
        # folder = Path(f"result_{dataset_name}_{folder_name_suffix}")
        # folder.mkdir(parents=True, exist_ok=True)
        # folder_name = folder.name

    fig, ax = add_hyperparameter_text(params)
    # plt.figure(figsize=(10, 5))
    ax.plot(train_loss_list, label='Train Loss', color='blue')
    ax.plot(valid_learning_loss_list, label='Valid Loss', color='orange')
    ax.plot(test_learning_loss_list, label='Test Loss', color='green')
    # ax.plot(valid_loss_list, label='Valid Loss', color='orange')
    # ax.plot(test_loss_list, label='Test Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.savefig('{}/loss_{}_{}.png'.format(folder_name, dataset_name, timestamp))
    # plt.clf()  # Clear the current figure for the next plot
    plt.close()
    # Plotting the acc in one figure
    fig, ax = add_hyperparameter_text(params)
    # plt.figure(figsize=(10, 5))
    ax.plot(train_accuracy_list, label='Train Accuracy', color='blue')
    ax.plot(valid_accuracy_list, label='Valid Accuracy', color='orange')
    ax.plot(test_accuracy_list, label='Test Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.savefig('{}/accuracy_{}_{}.png'.format(folder_name, dataset_name, timestamp))
    # plt.clf()  # Clear the current figure for the next plot
    plt.close()
    # Plotting the energy in one figure
    fig, ax = add_hyperparameter_text(params)
    # plt.figure(figsize=(10, 5))
    # ax.plot(energy_loss_list, label='Dirichlet Energy', color='blue')
    # plt.xlabel('Epochs')
    # plt.ylabel('Energy')
    # plt.title('Energy vs Epochs')
    # plt.legend()
    # plt.savefig('{}/energy_{}_{}.png'.format(folder_name, dataset_name, timestamp))
    # # plt.clf()  # Clear the current figure for the next plot
    # plt.close()

    # if save_model:
    #     torch.save(model.state_dict(), F'saved_models/model_weights_{dataset_name}_{num_mp_layers}_{mp_hidden_dim}_{num_fl_layers}_{fl_hidden_dim}_{dropout}.pth')

    # return best_val, best_test, model, train_accuracy_list, valid_accuracy_list, test_accuracy_list, energy_loss_list, \
    #         train_loss_list, valid_loss_list, test_loss_list, train_learning_loss_list, \
    #         valid_learning_loss_list, test_learning_loss_list, norms_list
    return best_val, best_test, model, train_accuracy_list, valid_accuracy_list, test_accuracy_list, \
            train_loss_list, valid_learning_loss_list, test_learning_loss_list



def main_experiment(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim=3000, fl_hidden_dim=3000, optimizer_lr=0.01,
                    loss_func='CrossEntropyLoss', total_epoch=500,
                    freeze=False, skip_connection=False, dropout=0, folder_name_suffix="", energy_lambda=1.0, energy_threshold=10000):
    # dropout_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # best_vals = np.zeros(len(dropout_rates))
    # best_tests = np.zeros(len(dropout_rates))
    # for i, dropout in enumerate(dropout_rates):
    #     best_val, best_test, _ = run_with_regularization(dataset_name, 'AdamW', weight_decay, num_mp_layers, num_fl_layers, mp_hidden_dim,
    #                                                      fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index=0,
    #                                                      freeze=freeze, dropout=dropout, skip_connection=skip_connection,
    #                                                      folder_name_suffix=folder_name_suffix)
    #     best_vals[i] = best_val
    #     best_tests[i] = best_test
    # num_mp_layers = 6
    best_val, best_test, _, train_accuracy_list, valid_accuracy_list, test_accuracy_list, \
    train_loss_list, valid_learning_loss_list, test_learning_loss_list = \
            run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim, fl_hidden_dim,
                optimizer_lr, loss_func, total_epoch,
                freeze=freeze, save_model=False, skip_connection=skip_connection, dropout=0,
                folder_name_suffix=folder_name_suffix, energy_lambda=energy_lambda, energy_threshold=energy_threshold)
    
    return best_val, best_test, train_accuracy_list, valid_accuracy_list, test_accuracy_list, train_loss_list, \
          valid_learning_loss_list, test_learning_loss_list

    # params = {
    #     'dataset_name': dataset_name,
    #     'mp_hidden_dim': mp_hidden_dim,
    #     'optimizer_lr': optimizer_lr,
    #     'freeze': freeze,
    #     'skip_connection': skip_connection
    # }
    # fig, ax = add_hyperparameter_text(params)
    # # Plot with evenly spaced points
    # label_prefix = f'mp_{num_mp_layers}_fc_{num_fl_layers}'
    # ax.plot(range(len(train_accuracy_list)), train_accuracy_list, label=f'{label_prefix} Train Accuracy', color='blue', linestyle='-')
    # ax.plot(range(len(valid_accuracy_list)), valid_accuracy_list, label=f'{label_prefix} Valid Accuracy', color='red', linestyle='-')
    # ax.plot(range(len(test_accuracy_list)), test_accuracy_list, label=f'{label_prefix} Test Accuracy', color='green', linestyle='-')

    # # ---- Clean up GPU memory ----
    # torch.cuda.empty_cache()       # release cached blocks
    # gc.collect()                   # force Python to collect garbage
    # torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1


    # num_mp_layers = 1
    # num_fl_layers = 2
    # best_val, best_test, _, train_accuracy_list, valid_accuracy_list, test_accuracy_list = \
    #         run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim,
    #             fl_hidden_dim, math.pi-3, optimizer_lr, loss_func, total_epoch, index=0,
    #             freeze=freeze, save_model=False, skip_connection=skip_connection, dropout=0,
    #             folder_name_suffix='undersmoothing_july21__meeting_qian2')
    # label_prefix = f'mp_{num_mp_layers}_fc_{num_fl_layers}'
    # ax.plot(range(len(train_accuracy_list)), train_accuracy_list, label=f'{label_prefix} Train Accuracy', color='orange', linestyle='--')
    # ax.plot(range(len(valid_accuracy_list)), valid_accuracy_list, label=f'{label_prefix} Valid Accuracy', color='purple', linestyle='--')
    # ax.plot(range(len(test_accuracy_list)), test_accuracy_list, label=f'{label_prefix} Test Accuracy', color='yellow', linestyle='--')


    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('accuracy vs epoch for different number of mp and fc')
    # plt.legend()
    # plt.savefig('{}/{}_undersmoothing_experiment_accuracy_{}.png'.format(folder_name, dataset_name, get_timestamp()))
    # plt.clf()  # Clear the current figure for the next plot


# Create folder for results
folder = Path(f"result_overuniquess")
folder.mkdir(parents=True, exist_ok=True)
folder_name = folder.name


dataset_name = 'citeseer'
num_mp_layers = 1
num_fl_layers = 1
mp_hidden_dim = 512
fl_hidden_dim = 512
optimizer_lr = 0.001
freeze = False
skip_connection = True
layer_pairs = [[0,4], [1,3], [2,2], [3,1], [4,0]]
# lambdas = [0]
# lambdas = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
total_epoch=500
all_energy_loss_list = []
all_train_loss = []
all_valid_loss = []
all_test_loss = []
all_train_acc = []
all_valid_acc = []
all_test_acc = []
best_vals = []
best_tests = []
all_train_learning_loss = []
all_valid_learning_loss = []
all_test_learning_loss = []
all_norms = []

for num_mp_layers, num_fl_layers in layer_pairs:
    best_val, best_test, train_accuracy_list, valid_accuracy_list, test_accuracy_list, train_loss_list, valid_loss_list, \
    test_loss_list = \
    main_experiment(dataset_name=dataset_name,
                    num_mp_layers=num_mp_layers,
                    num_fl_layers=num_fl_layers,
                    mp_hidden_dim=mp_hidden_dim,
                    fl_hidden_dim=fl_hidden_dim,
                    optimizer_lr=optimizer_lr,
                    loss_func='CrossEntropyLoss', total_epoch=total_epoch,
                    freeze=freeze, skip_connection=skip_connection, folder_name_suffix="", energy_lambda=0, energy_threshold=-1) #0.00001
    # all_energy_loss_list.append(energy_loss_list)
    all_train_loss.append(train_loss_list)
    all_valid_loss.append(valid_loss_list)
    all_test_loss.append(test_loss_list)
    all_train_acc.append(train_accuracy_list)
    all_valid_acc.append(valid_accuracy_list)
    all_test_acc.append(test_accuracy_list)
    best_tests.append(best_test)
    best_vals.append(best_val)
    # all_train_learning_loss.append(train_learning_loss_list)
    # all_valid_learning_loss.append(valid_learning_loss_list)
    # all_test_learning_loss.append(test_learning_loss_list)
    # all_norms.append(norms_list)
    # ---- Clean up GPU memory ----
    torch.cuda.empty_cache()       # release cached blocks
    gc.collect()                   # force Python to collect garbage
    torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1


params = {
    'dataset_name': dataset_name,
    'num_mp_layers': num_mp_layers,
    'num_fl_layers': num_fl_layers,
    'mp_hidden_dim': mp_hidden_dim,
    'fl_hidden_dim': fl_hidden_dim,
    'optimizer_lr': optimizer_lr,
    'freeze': freeze,
    'skip_connection': skip_connection
}

# optional: plot norms for pool scaling checking
# num_layers = num_mp_layers + 1
# folder_name = 'norm_checking_with_batch_normalization'
# for i in range(len(lambdas)):
#     norms_list = np.array(all_norms[i], dtype=np.float32)
#     fig, ax = add_hyperparameter_text(params)
#     for j in range(num_layers):
#         label_prefix = f'norm_at_layer_{j}_with_energy_lambda_{lambdas[i]}'
#         ax.plot(norms_list[:,j], label=f'{label_prefix}', linestyle='-')
#     plt.xlabel('Epoch')
#     plt.ylabel('Norm')
#     plt.legend()
#     plt.savefig('{}/{}_norm_plot_with_lambda_{}.png'.format(folder_name, dataset_name, lambdas[i]))
#     plt.clf()  # Clear the current figure for the next plot

# sys.exit()

# # first plot: energy loss across different lambda values
# fig, ax = add_hyperparameter_text(params)
# # Plot with evenly spaced points
# for i, energy_loss_list in enumerate(all_energy_loss_list):
#     label_prefix = f'energy_lambda_{lambdas[i]}'
#     ax.plot(energy_loss_list, label=f'{label_prefix} Energy', linestyle='-')
# plt.xlabel('Epoch')
# plt.ylabel('Energy')
# plt.legend()
# plt.savefig('{}/{}_energy_loss_with_different_lambda_{}.png'.format(folder_name, dataset_name, get_timestamp()))
# plt.clf()  # Clear the current figure for the next plot


# second plot: train loss across different lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, train_loss_list in enumerate(all_train_loss):
    label_prefix = f'mp_{layer_pairs[i][0]}_fc_{layer_pairs[i][1]}'
    ax.plot(train_loss_list, label=f'{label_prefix} train loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.legend()
plt.savefig('{}/{}_train_loss_with_different_layers_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot


# label_prefix = f'mp_{num_mp_layers}_fc_{num_fl_layers}'
# ax.plot(range(len(train_accuracy_list)), train_accuracy_list, label=f'{label_prefix} Train Accuracy', color='orange', linestyle='--')
# ax.plot(range(len(valid_accuracy_list)), valid_accuracy_list, label=f'{label_prefix} Valid Accuracy', color='purple', linestyle='--')
# ax.plot(range(len(test_accuracy_list)), test_accuracy_list, label=f'{label_prefix} Test Accuracy', color='yellow', linestyle='--')

# # third plot: valid loss across different lambda values
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, valid_loss_list in enumerate(all_valid_loss):
    label_prefix = f'mp_{layer_pairs[i][0]}_fc_{layer_pairs[i][1]}'
    ax.plot(valid_loss_list, label=f'{label_prefix} valid loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Valid Loss')
plt.legend()
plt.savefig('{}/{}_valid_loss_with_different_layers_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot


# # fourth plot: test loss across different lambda values
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, test_loss_list in enumerate(all_test_loss):
    label_prefix = f'mp_{layer_pairs[i][0]}_fc_{layer_pairs[i][1]}'
    ax.plot(test_loss_list, label=f'{label_prefix} test loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.legend()
plt.savefig('{}/{}_test_loss_with_different_layers_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot



# fifth plot: test accuracy across different lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, test_accuracy_list in enumerate(all_test_acc):
    label_prefix = f'mp_{layer_pairs[i][0]}_fc_{layer_pairs[i][1]}'
    ax.plot(test_accuracy_list, label=f'{label_prefix} test accuracy', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend()
plt.savefig('{}/{}_test_accuracy_with_different_layers_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot


# # sixth plot: train accuracy across different lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, train_accuracy_list in enumerate(all_train_acc):
    label_prefix = f'mp_{layer_pairs[i][0]}_fc_{layer_pairs[i][1]}'
    ax.plot(train_accuracy_list, label=f'{label_prefix} train accuracy', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.legend()
plt.savefig('{}/{}_train_accuracy_with_different_layers_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot

# seventh plot: valid accuracy across different lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, valid_accuracy_list in enumerate(all_valid_acc):
    label_prefix = f'mp_{layer_pairs[i][0]}_fc_{layer_pairs[i][1]}'
    ax.plot(valid_accuracy_list, label=f'{label_prefix} valid accuracy', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Valid Accuracy')
plt.legend()
plt.savefig('{}/{}_valid_accuracy_with_different_layers_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot

# eighth plot: best test, best val across diffferent lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
ax.plot(range(len(layer_pairs)), best_vals, label=f'Best valid accuracy', linestyle='-', marker='o')
ax.plot(range(len(layer_pairs)), best_tests, label=f'Best test accuracy', linestyle='-', marker='o')

ax.set_xticks(range(len(layer_pairs)))
xlables = [f'mp_{layer_pairs[i][0]}_fc_{layer_pairs[i][1]}' for i in range(len(layer_pairs))]
ax.set_xticklabels(xlables)

plt.xlabel('Layer pairs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('{}/{}_best_accuracy_with_different_layers_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot

# # nineth plot: train learning loss across different lambda
# fig, ax = add_hyperparameter_text(params)
# # Plot with evenly spaced points
# for i, train_learning_loss_list in enumerate(all_train_learning_loss):
#     label_prefix = f'energy_lambda_{lambdas[i]}'
#     ax.plot(train_learning_loss_list, label=f'{label_prefix} train learning loss', linestyle='-')
# # ax.set_yscale("log")
# plt.xlabel('Epoch')
# plt.ylabel('Train Learning Loss')
# plt.legend()
# plt.savefig('{}/{}_train_learning_loss_with_different_lambda_{}.png'.format(folder_name, dataset_name, get_timestamp()))
# plt.clf()  # Clear the current figure for the next plot


# # tenth plot: train learning loss across different lambda
# fig, ax = add_hyperparameter_text(params)
# # Plot with evenly spaced points
# for i, valid_learning_loss_list in enumerate(all_valid_learning_loss):
#     label_prefix = f'energy_lambda_{lambdas[i]}'
#     ax.plot(valid_learning_loss_list, label=f'{label_prefix} valid learning loss', linestyle='-')
# plt.xlabel('Epoch')
# plt.ylabel('Valid Learning Loss')
# plt.legend()
# plt.savefig('{}/{}_valid_learning_loss_with_different_lambda_{}.png'.format(folder_name, dataset_name, get_timestamp()))
# plt.clf()  # Clear the current figure for the next plot


# # nineth plot: train learning loss across different lambda
# fig, ax = add_hyperparameter_text(params)
# # Plot with evenly spaced points
# for i, test_learning_loss_list in enumerate(all_test_learning_loss):
#     label_prefix = f'energy_lambda_{lambdas[i]}'
#     ax.plot(test_learning_loss_list, label=f'{label_prefix} test learning loss', linestyle='-')
# plt.xlabel('Epoch')
# plt.ylabel('Test Learning Loss')
# plt.legend()
# plt.savefig('{}/{}_test_learning_loss_with_different_lambda_{}.png'.format(folder_name, dataset_name, get_timestamp()))
# plt.clf()  # Clear the current figure for the next plot

