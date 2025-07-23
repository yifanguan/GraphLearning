import torch
# from torch_geometric.loader import DataLoader
# from torch_geometric.utils import degree
# from ogb.graphproppred import PygGraphPropPredDataset
# from ogb.graphproppred import Evaluator
from models.dln import DecoupleModel, iGNN, iGNN_V2, iGNN_energy_version
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
# from utils.split import split_dataset
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
from utils.dataset import load_dataset
from utils.data_split_util import rand_train_test_idx
from utils.timestamp import get_timestamp
from torch_geometric.utils import to_undirected
from utils.over_smoothing_measure import dirichlet_energy

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



def train(model, data, train_idx, optimizer, criterion):
    model.train()
    # out = model(data.x, data.edge_index)
    out = model(data)
    # is_labeled = data.y == data.y
    energy_loss = dirichlet_energy(data.x[train_idx], data.edge_index)
    loss = criterion(out[train_idx], data.y[train_idx]) + energy_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, criterion, data, train_split_idx, validation_split_idx, test_split_idx):
    model.eval()
    out = model(data)
    # train_loss = criterion(out[train_split_idx], data.y[train_split_idx])
    energy_loss = dirichlet_energy(data.x, data.edge_index)
    valid_loss = criterion(out[validation_split_idx], data.y[validation_split_idx])
    test_loss = criterion(out[test_split_idx], data.y[test_split_idx])

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

    return train_acc, valid_acc, test_acc, valid_loss.item(), test_loss.item()


def run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim, fl_hidden_dim,
        epsilon, optimizer_lr, loss_func, total_epoch, index, freeze, save_model=False, skip_connection=False, dropout=0,
        folder_name_suffix=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    display_step = 20
    data = load_dataset(data_dir='data', dataset_name=dataset_name).to(device)
    data.edge_index = to_undirected(data.edge_index)
    d = data.x.shape[1]
    c = max(data.y.max().item() + 1, data.y.shape[0])

    # data split for train, val, and test
    if hasattr(data, 'train_mask'):
        train_idx = torch.where(data.train_mask)[0]
        valid_idx = torch.where(data.val_mask)[0]
        test_idx = torch.where(data.test_mask)[0]
    else:
        train_idx, valid_idx, test_idx = rand_train_test_idx(data.y, train_prop=0.6, valid_prop=0.2)

    model = iGNN_energy_version (
        in_dim=d,
        out_dim=c,
        mp_width=mp_hidden_dim,
        num_mp_layers = num_mp_layers,
        freeze=freeze,
        skip_connection=skip_connection,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_lr)
    criterion = torch.nn.CrossEntropyLoss()
    if loss_func == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_func == 'NLLLoss':
        criterion = torch.nn.NLLLoss()

    print('Experiment run {}'.format(index))
    print(f'dataset: {dataset_name}')
    print(f'num_mp_layers: {num_mp_layers}')
    print(f'mp_hidden_dim: {mp_hidden_dim}')
    print(f'optimizer_lr: {optimizer_lr}')
    print(f'loss_func: {loss_func}')
    print(f'total_epoch: {total_epoch}')


    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []
    best_val = float('-inf')
    best_test = float('-inf')
    train_accuracy_list = []
    valid_accuracy_list = []
    test_accuracy_list = []

    for epoch in tqdm(range(1,total_epoch)):
        loss = train(model,data,train_idx,optimizer,criterion)
        train_acc, valid_acc, test_acc, valid_loss, test_loss = evaluate(model, criterion, data, train_idx, valid_idx, test_idx)

        train_accuracy_list.append(train_acc)
        valid_accuracy_list.append(valid_acc)
        test_accuracy_list.append(test_acc)
        train_loss_list.append(loss)
        valid_loss_list.append(valid_loss)
        test_loss_list.append(test_loss)

        if test_acc > best_test:
            best_test = test_acc
        if valid_acc > best_val:
            best_val = valid_acc

        if epoch % display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}%, '
                f'Test: {100 * test_acc:.2f}%, '
                f'Best Valid: {100 * best_val:.2f}%, '
                f'Best Test: {100 * best_test:.2f}%')

    with open('experiment_records.txt', 'a') as f:
        json.dump(train_accuracy_list, f)
        f.write("\n")
        json.dump(valid_accuracy_list, f)
        f.write("\n")
        json.dump(test_accuracy_list, f)
        f.write("\n")
        f.write('best Valid: ' + str(best_val) + '\n')
        f.write('best test: ' + str(best_test) + '\n')
        f.write("\n\n")

    print(f'train_accuracy_list: {train_accuracy_list}')
    print(f'valid_accuracy_list: {valid_accuracy_list}')
    print(f'test_accuracy_list: {test_accuracy_list}')
    print(f'best validation: {best_val}')
    print(f'best test: {best_test}')

    # Plotting the loss, min cell is 0.1, large figure
    params = {
        'dataset_name': dataset_name,
        'num_mp_layers': num_mp_layers,
        'num_fl_layers': num_fl_layers,
        'mp_hidden_dim': mp_hidden_dim,
        'fl_hidden_dim': fl_hidden_dim,
        'optimizer_lr': optimizer_lr,
        'freeze': freeze,
        'skip_connection': skip_connection,
        'dropout': dropout
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
        folder = Path(f"result_{dataset_name}_{folder_name_suffix}")
        folder.mkdir(parents=True, exist_ok=True)
        folder_name = folder.name

    fig, ax = add_hyperparameter_text(params)
    # plt.figure(figsize=(10, 5))
    ax.plot(train_loss_list, label='Train Loss', color='blue')
    ax.plot(valid_loss_list, label='Valid Loss', color='orange')
    ax.plot(test_loss_list, label='Test Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.savefig('{}/loss_{}_{}_{}.png'.format(folder_name, dataset_name, index, timestamp))
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
    plt.savefig('{}/accuracy_{}_{}_{}.png'.format(folder_name, dataset_name, index, timestamp))
    # plt.clf()  # Clear the current figure for the next plot
    plt.close()

    if save_model:
        torch.save(model.state_dict(), F'saved_models/model_weights_{dataset_name}_{num_mp_layers}_{mp_hidden_dim}_{num_fl_layers}_{fl_hidden_dim}_{dropout}.pth')

    return best_val, best_test, model, train_accuracy_list, valid_accuracy_list, test_accuracy_list


# For filename
timestamp = get_timestamp()

# Create folder for results
folder = Path(f"result_{timestamp}")
folder.mkdir(parents=True, exist_ok=True)
folder_name = folder.name




def undersmoothing_experiment(dataset_name, mp_hidden_dim=3000, fl_hidden_dim=512, optimizer_lr=0.01,
                                loss_func='CrossEntropyLoss', total_epoch=500,
                                freeze=False, skip_connection=False, folder_name_suffix=""):
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
    num_mp_layers = 6
    num_fl_layers = 2
    best_val, best_test, _, train_accuracy_list, valid_accuracy_list, test_accuracy_list = \
            run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim,
                fl_hidden_dim, math.pi-3, optimizer_lr, loss_func, total_epoch, index=0,
                freeze=freeze, save_model=False, skip_connection=skip_connection, dropout=0,
                folder_name_suffix='undersmoothing_july21_meeting_qian')

    params = {
        'dataset_name': dataset_name,
        'mp_hidden_dim': mp_hidden_dim,
        'fl_hidden_dim' : fl_hidden_dim,
        'optimizer_lr': optimizer_lr,
        'freeze': freeze,
        'skip_connection': skip_connection
    }
    fig, ax = add_hyperparameter_text(params)
    # Plot with evenly spaced points
    label_prefix = f'mp_{num_mp_layers}_fc_{num_fl_layers}'
    ax.plot(range(len(train_accuracy_list)), train_accuracy_list, label=f'{label_prefix} Train Accuracy', color='blue', linestyle='-')
    ax.plot(range(len(valid_accuracy_list)), valid_accuracy_list, label=f'{label_prefix} Valid Accuracy', color='red', linestyle='-')
    ax.plot(range(len(test_accuracy_list)), test_accuracy_list, label=f'{label_prefix} Test Accuracy', color='green', linestyle='-')

    # ---- Clean up GPU memory ----
    torch.cuda.empty_cache()       # release cached blocks
    gc.collect()                   # force Python to collect garbage
    torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1


    num_mp_layers = 1
    num_fl_layers = 2
    best_val, best_test, _, train_accuracy_list, valid_accuracy_list, test_accuracy_list = \
            run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim,
                fl_hidden_dim, math.pi-3, optimizer_lr, loss_func, total_epoch, index=0,
                freeze=freeze, save_model=False, skip_connection=skip_connection, dropout=0,
                folder_name_suffix='undersmoothing_july21__meeting_qian2')
    label_prefix = f'mp_{num_mp_layers}_fc_{num_fl_layers}'
    ax.plot(range(len(train_accuracy_list)), train_accuracy_list, label=f'{label_prefix} Train Accuracy', color='orange', linestyle='--')
    ax.plot(range(len(valid_accuracy_list)), valid_accuracy_list, label=f'{label_prefix} Valid Accuracy', color='purple', linestyle='--')
    ax.plot(range(len(test_accuracy_list)), test_accuracy_list, label=f'{label_prefix} Test Accuracy', color='yellow', linestyle='--')


    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('accuracy vs epoch for different number of mp and fc')
    plt.legend()
    plt.savefig('{}/{}_undersmoothing_experiment_accuracy_{}.png'.format(folder_name, dataset_name, get_timestamp()))
    plt.clf()  # Clear the current figure for the next plot



# Create folder for results
folder = Path(f"result_undersmoothing")
folder.mkdir(parents=True, exist_ok=True)
folder_name = folder.name


undersmoothing_experiment(dataset_name='cora',
                            mp_hidden_dim=4000, fl_hidden_dim=128,
                            optimizer_lr=0.001,
                            loss_func='CrossEntropyLoss', total_epoch=500,
                            freeze=False, skip_connection=False, folder_name_suffix="")
# ---- Clean up GPU memory ----
# torch.cuda.empty_cache()       # release cached blocks
# gc.collect()                   # force Python to collect garbage
# torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1
