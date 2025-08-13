import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
# from torch_geometric.loader import DataLoader
# from torch_geometric.utils import degree
# from ogb.graphproppred import PygGraphPropPredDataset
# from ogb.graphproppred import Evaluator
from models.dln import DecoupleModel, iGNN, iGNN_V2, iGNN_energy_version, iGNN_energy_version_fc, iGNN_tune_version
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
from utils.dataset import load_dataset, load_large_dataset
from utils.data_split_util import rand_train_test_idx
from utils.timestamp import get_timestamp
from torch_geometric.utils import to_undirected, add_self_loops
# from utils.over_smoothing_measure import dirichlet_energy, normalized_dirichlet_energy
import gc

# TODO: improve result folder structure
# TODO: dashed line for std result after running multiple runs

def tensor_size_MB(tensor):
    return tensor.element_size() * tensor.nelement() / 1e6

def log_gpu_tensors(tag=""):
    print(f"\n🔍 GPU tensors at {tag}:")
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size = tensor_size_MB(obj)
                print(f"{type(obj)} {obj.size()} | {size:.2f} MB")
                total += size
        except Exception:
            pass
    print(f"🔢 Total GPU tensor memory: {total:.2f} MB\n")

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def train(model, data, train_idx, optimizer, criterion, energy_lambda, energy_threshold):
    model.train()
    out, embedding, norms_per_layer = model(data)
    energy_loss = torch.tensor(0).cuda()
    norm_energy_loss = torch.tensor(0).cuda()
    loss = criterion(out[train_idx], data.y[train_idx])
    train_learning_loss = loss
    # if energy_loss > energy_threshold:
    #     loss += energy_loss * energy_lambda
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), energy_loss.item(), train_learning_loss.item(), norms_per_layer, norm_energy_loss.item()


@torch.no_grad()
def evaluate(model, criterion, data, train_split_idx, validation_split_idx, test_split_idx, energy_lambda):
    model.eval()
    out, embedding, _ = model(data)
    energy_loss = torch.tensor(0).cuda()
    valid_learning_loss = criterion(out[validation_split_idx], data.y[validation_split_idx])
    valid_loss = valid_learning_loss + energy_loss * energy_lambda
    test_learning_loss = criterion(out[test_split_idx], data.y[test_split_idx])
    test_loss = test_learning_loss + energy_loss * energy_lambda

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

    return train_acc, valid_acc, test_acc, valid_loss.item(), test_loss.item(), energy_loss.item(), \
            valid_learning_loss.item(), test_learning_loss.item()


def run(dataset_name, num_mp_layers, mp_hidden_dim, num_fl_layers, fl_hidden_dim, num_mp_smoothing_layers,
        optimizer_lr, loss_func, total_epoch, freeze, save_model=False, skip_connection=False, dropout=0,
        folder_name_suffix="", energy_lambda=1.0, energy_threshold=10000):
    # fix_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    display_step = 5
    dataset = load_large_dataset(data_dir='data', name=dataset_name)

    d = dataset.graph.x.shape[1]
    c = dataset.label.max().item() + 1
    n = dataset.graph.x.shape[0]
    assert c == dataset.num_classes
    print(f'Dataset: {dataset_name}, num nodes: {n}, num node features: {d}, num classes: {c}')
    dataset.graph.edge_index = to_undirected(dataset.graph.edge_index)
    dataset.graph.edge_index, _ = add_self_loops(dataset.graph.edge_index, num_nodes=n)

    dataset.graph = dataset.graph.to(device)

    # data split for train, val, and test
    # if hasattr(data, 'train_mask'):
    #     train_idx = torch.where(data.train_mask)[0]
    #     valid_idx = torch.where(data.val_mask)[0]
    #     test_idx = torch.where(data.test_mask)[0]

    #     # unlabeled_mask = ~(data.train_mask | data.val_mask | data.test_mask)
    #     # unlabeled_idx = torch.where(unlabeled_mask)[0]
    #     # num_samples = int(len(unlabeled_idx) * extra_train_data_rate)

    #     # Randomly shuffle and select a portion
    #     # selected_unlabeled_idx = unlabeled_idx[torch.randperm(len(unlabeled_idx))[:num_samples]]
    #     # train_idx = torch.cat([train_idx, unlabeled_idx])
    # else:
    #     train_idx, valid_idx, test_idx = rand_train_test_idx(data.y, train_prop=0.6, valid_prop=0.2)
    train_idx = dataset.train_idx
    valid_idx = dataset.valid_idx
    test_idx = dataset.test_idx

    model = iGNN_tune_version (
        in_dim=d,
        num_mp_layers=num_mp_layers,
        mp_width=d,
        num_fl_layers=num_fl_layers,
        fl_width=fl_hidden_dim,
        out_dim=c,
        freeze=freeze,
        skip_connection=skip_connection
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
    # print(f'mp_hidden_dim: {mp_hidden_dim}')
    print(f'num_fl_layers: {num_fl_layers}')
    print(f'fl_width: {fl_hidden_dim}')
    print(f'num_mp_layers_after_fl: {num_mp_smoothing_layers}')
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
    norm_energy_loss_list = []

    for epoch in tqdm(range(1,total_epoch)):
        loss, energy_loss, train_learning_loss, norms_per_layer, norm_energy_loss = train(model,dataset.graph,train_idx,optimizer,criterion, energy_lambda, energy_threshold)
        train_acc, valid_acc, test_acc, valid_loss, test_loss, _, valid_learning_loss, test_learning_loss  = evaluate(model, criterion, dataset.graph, train_idx, valid_idx, test_idx, energy_lambda)


        train_accuracy_list.append(train_acc)
        valid_accuracy_list.append(valid_acc)
        test_accuracy_list.append(test_acc)
        train_loss_list.append(loss)
        valid_loss_list.append(valid_loss)
        test_loss_list.append(test_loss)
        energy_loss_list.append(energy_loss)
        norm_energy_loss_list.append(norm_energy_loss)
        train_learning_loss_list.append(train_learning_loss)
        valid_learning_loss_list.append(valid_learning_loss)
        test_learning_loss_list.append(test_learning_loss)
        norms_list.append(norms_per_layer)

        if test_acc > best_test:
            best_test = test_acc
        if valid_acc > best_val:
            best_val = valid_acc

        if epoch % display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Learning Loss: {train_learning_loss:.4f}, '
                f'Energy: {energy_loss:.4f}, '
                f'Normalized Energy: {norm_energy_loss:.4f}, '
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
        # 'mp_hidden_dim': mp_hidden_dim,
        'num_fl_layers': num_fl_layers,
        'fl_hidden_dim': fl_hidden_dim,
        'num_mp_layers_after_fl': num_mp_smoothing_layers,
        'optimizer_lr': optimizer_lr,
        'freeze': freeze,
        'skip_connection': skip_connection,
        # 'dropout': dropout,
        'energy lambda': energy_lambda
    }
    
    # in case run() is executed in other files other than main.py
    try:
        timestamp
    except NameError:
        timestamp = get_timestamp()
    try:
        folder_name
    except NameError:
        # Create folder for results result_simple_mp_after_fl
        folder = Path(f"result_loss_tune")
        folder.mkdir(parents=True, exist_ok=True)
        folder_name = folder.name
        # folder = Path(f"result_{dataset_name}_{folder_name_suffix}")
        # folder.mkdir(parents=True, exist_ok=True)
        # folder_name = folder.name

    fig, ax = add_hyperparameter_text(params)
    # plt.figure(figsize=(10, 5))
    ax.plot(train_loss_list, label='Train Loss', color='blue')
    ax.plot(valid_loss_list, label='Valid Loss', color='orange')
    ax.plot(test_loss_list, label='Test Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss vs Epochs')
    ax.set_yscale("log")
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
    ax.plot(norm_energy_loss_list, label='Normalized Dirichlet Energy', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Energy (log scale)')
    plt.title('Energy vs Epochs')
    ax.set_yscale("log")
    plt.legend()
    plt.savefig('{}/energy_{}_{}.png'.format(folder_name, dataset_name, timestamp))
    # plt.clf()  # Clear the current figure for the next plot
    plt.close()

    # if save_model:
    #     torch.save(model.state_dict(), F'saved_models/model_weights_{dataset_name}_{num_mp_layers}_{mp_hidden_dim}_{num_fl_layers}_{fl_hidden_dim}_{dropout}.pth')

    return best_val, best_test, model, train_accuracy_list, valid_accuracy_list, test_accuracy_list, energy_loss_list, \
            train_loss_list, valid_loss_list, test_loss_list, train_learning_loss_list, \
            valid_learning_loss_list, test_learning_loss_list, norms_list, norm_energy_loss_list




def main_experiment(dataset_name, num_mp_layers, mp_hidden_dim=3000, num_fl_layers=3, fl_hidden_dim=512, num_mp_smoothing_layers=1,
                    optimizer_lr=0.01, loss_func='CrossEntropyLoss', total_epoch=500,
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
    best_val, best_test, _, train_accuracy_list, valid_accuracy_list, test_accuracy_list, energy_loss_list, \
    train_loss_list, valid_loss_list, test_loss_list, train_learning_loss_list, valid_learning_loss_list, \
    test_learning_loss_list, norms_list, norm_energy_loss_list = \
            run(dataset_name, num_mp_layers, mp_hidden_dim, num_fl_layers, fl_hidden_dim, num_mp_smoothing_layers,
                optimizer_lr, loss_func, total_epoch,
                freeze=freeze, save_model=False, skip_connection=skip_connection, dropout=0,
                folder_name_suffix=folder_name_suffix, energy_lambda=energy_lambda, energy_threshold=energy_threshold)
    
    return best_val, best_test, train_accuracy_list, valid_accuracy_list, test_accuracy_list, energy_loss_list, train_loss_list, valid_loss_list, test_loss_list, \
            train_learning_loss_list, valid_learning_loss_list, test_learning_loss_list, norms_list, norm_energy_loss_list

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





# main_experiment(dataset_name='cora',
#                 num_mp_layers=6,
#                 mp_hidden_dim=4000,
#                 optimizer_lr=0.001,
#                 loss_func='CrossEntropyLoss', total_epoch=500,
#                 freeze=False, skip_connection=False, folder_name_suffix="", energy_lambda=1.0)


# main_experiment(dataset_name='cora',
#                 num_mp_layers=3,
#                 mp_hidden_dim=4000,
#                 optimizer_lr=0.001,
#                 loss_func='CrossEntropyLoss', total_epoch=1200,
#                 freeze=False, skip_connection=False, folder_name_suffix="", energy_lambda=0.01)
# # ---- Clean up GPU memory ----
# torch.cuda.empty_cache()       # release cached blocks
# gc.collect()                   # force Python to collect garbage
# torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1


# main_experiment(dataset_name='cora',
#                 num_mp_layers=3,
#                 mp_hidden_dim=4000,
#                 optimizer_lr=0.001,
#                 loss_func='CrossEntropyLoss', total_epoch=1200,
#                 freeze=False, skip_connection=False, folder_name_suffix="", energy_lambda=0.001)
# # ---- Clean up GPU memory ----
# torch.cuda.empty_cache()       # release cached blocks
# gc.collect()                   # force Python to collect garbage
# torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1


# main_experiment(dataset_name='cora',
#                 num_mp_layers=3,
#                 mp_hidden_dim=4000,
#                 optimizer_lr=0.001,
#                 loss_func='CrossEntropyLoss', total_epoch=1200,
#                 freeze=False, skip_connection=False, folder_name_suffix="", energy_lambda=0)
# # ---- Clean up GPU memory ----
# torch.cuda.empty_cache()       # release cached blocks
# gc.collect()                   # force Python to collect garbage
# torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1


# main_experiment(dataset_name='cora',
#                 num_mp_layers=2,
#                 mp_hidden_dim=4000,
#                 optimizer_lr=0.001,
#                 loss_func='CrossEntropyLoss', total_epoch=1200,
#                 freeze=False, skip_connection=False, folder_name_suffix="", energy_lambda=0.01)
# # ---- Clean up GPU memory ----
# torch.cuda.empty_cache()       # release cached blocks
# gc.collect()                   # force Python to collect garbage
# torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1


# main_experiment(dataset_name='cora',
#                 num_mp_layers=4,
#                 mp_hidden_dim=4000,
#                 optimizer_lr=0.001,
#                 loss_func='CrossEntropyLoss', total_epoch=1200,
#                 freeze=False, skip_connection=False, folder_name_suffix="", energy_lambda=0.01)
# # ---- Clean up GPU memory ----
# torch.cuda.empty_cache()       # release cached blocks
# gc.collect()                   # force Python to collect garbage
# torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1

# Create folder for results
folder = Path(f"result_tune")
folder.mkdir(parents=True, exist_ok=True)
folder_name = folder.name


dataset_name = 'ogbn-arxiv'
num_mp_layers = 2
mp_hidden_dim = 10
num_fl_layers = 1
fl_hidden_dim = 512
num_mp_smoothing_layers = 0
optimizer_lr = 0.01
freeze = False
skip_connection = False
# lambdas = [0]
# lambdas = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
extra_layers = [0, 1, 2]
total_epoch=3000
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
all_norm_energy_loss_list = []
label = 'extra_mp'

# for num_mp_smoothing_layers in extra_layers:
#     best_val, best_test, train_accuracy_list, valid_accuracy_list, test_accuracy_list, energy_loss_list, train_loss_list, valid_loss_list, \
#     test_loss_list, train_learning_loss_list, valid_learning_loss_list, test_learning_loss_list, norms_list, norm_energy_loss_list = \
#     main_experiment(dataset_name=dataset_name,
#                     num_mp_layers=num_mp_layers,
#                     mp_hidden_dim=mp_hidden_dim,
#                     num_fl_layers=num_fl_layers,
#                     fl_hidden_dim=fl_hidden_dim,
#                     num_mp_smoothing_layers=num_mp_smoothing_layers,
#                     optimizer_lr=optimizer_lr,
#                     loss_func='CrossEntropyLoss', total_epoch=total_epoch,
#                     freeze=freeze, skip_connection=skip_connection, folder_name_suffix="", energy_lambda=0, energy_threshold=-1) #0.00001
#     all_energy_loss_list.append(energy_loss_list)
#     all_train_loss.append(train_loss_list)
#     all_valid_loss.append(valid_loss_list)
#     all_test_loss.append(test_loss_list)
#     all_train_acc.append(train_accuracy_list)
#     all_valid_acc.append(valid_accuracy_list)
#     all_test_acc.append(test_accuracy_list)
#     best_tests.append(best_test)
#     best_vals.append(best_val)
#     all_train_learning_loss.append(train_learning_loss_list)
#     all_valid_learning_loss.append(valid_learning_loss_list)
#     all_test_learning_loss.append(test_learning_loss_list)
#     all_norms.append(norms_list)
#     all_norm_energy_loss_list.append(norm_energy_loss_list)
#     # ---- Clean up GPU memory ----
#     torch.cuda.empty_cache()       # release cached blocks
#     gc.collect()                   # force Python to collect garbage
#     torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1


import optuna

def objective(trial):
    # 超参数搜索空间
    num_mp_layers = trial.suggest_int("num_mp_layers", 0, 4)
    num_fl_layers = 1
    fl_hidden_dim = 512
    optimizer_lr = trial.suggest_loguniform("optimizer_lr", 1e-4, 1e-1)

    # 调用主函数
    best_val, best_test, *_ = main_experiment(
        dataset_name='ogbn-arxiv',
        num_mp_layers=num_mp_layers,
        num_fl_layers=num_fl_layers,
        fl_hidden_dim=fl_hidden_dim,
        optimizer_lr=optimizer_lr,
        total_epoch=1000,   # 为了快速调参可以减少 epoch
        energy_lambda=0.0,  # or set to something meaningful
        energy_threshold=-1,
        freeze=False,
        skip_connection=False
    )

    return best_test  # 以 validation accuracy 为优化目标

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # 可改为更大的 n_trials

print("🎯 Best trial:")
print(study.best_trial)

sys.exit()

params = {
    'dataset_name': dataset_name,
    'num_mp_layers': num_mp_layers,
    # 'mp_hidden_dim': mp_hidden_dim,
    'num_fl_layers': num_fl_layers,
    'fl_hidden_dim': fl_hidden_dim,
    'num_mp_layers_after_fl': num_mp_smoothing_layers,
    'optimizer_lr': optimizer_lr,
    'freeze': freeze,
    'skip_connection': skip_connection
}

# optional: plot norms for pool scaling checking
for i in range(len(extra_layers)):
    norms_list = np.array(all_norms[i], dtype=np.float32)
    fig, ax = add_hyperparameter_text(params)
    num_layers = num_mp_layers + num_fl_layers + extra_layers[i]
    for j in range(num_layers):
        label_prefix = f'norm_at_layer_{j}'
        ax.plot(norms_list[:,j], label=f'{label_prefix}', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Norm (log scale)')
    ax.set_yscale("log")
    plt.legend()
    plt.savefig('{}/{}_norm_plot_with_extra_mp_{}.png'.format(folder_name, dataset_name, extra_layers[i]))
    plt.clf()  # Clear the current figure for the next plot
    plt.close(fig)


# first plot: energy loss across different lambda values
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, norm_energy_loss_list in enumerate(all_norm_energy_loss_list):
    label_prefix = f'extra_mp_{extra_layers[i]}'
    ax.plot(norm_energy_loss_list, label=f'{label_prefix} normalized Energy', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Normalized Energy')
ax.set_yscale("log")
plt.legend()
plt.savefig('{}/{}_normalized_energy_loss_with_different_extra_mp_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot
plt.close(fig)

fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, energy_loss_list in enumerate(all_energy_loss_list):
    label_prefix = f'extra_mp_{extra_layers[i]}'
    ax.plot(energy_loss_list, label=f'{label_prefix} Energy', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Energy')
ax.set_yscale("log")
plt.legend()
plt.savefig('{}/{}_energy_loss_with_different_extra_mp_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot
plt.close(fig)

# second plot: train loss across different lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, train_loss_list in enumerate(all_train_loss):
    label_prefix = f'extra_mp_{extra_layers[i]}'
    ax.plot(train_loss_list, label=f'{label_prefix} train loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
ax.set_yscale("log")
plt.legend()
plt.savefig('{}/{}_train_loss_with_different_extra_mp_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot
plt.close(fig)

# label_prefix = f'mp_{num_mp_layers}_fc_{num_fl_layers}'
# ax.plot(range(len(train_accuracy_list)), train_accuracy_list, label=f'{label_prefix} Train Accuracy', color='orange', linestyle='--')
# ax.plot(range(len(valid_accuracy_list)), valid_accuracy_list, label=f'{label_prefix} Valid Accuracy', color='purple', linestyle='--')
# ax.plot(range(len(test_accuracy_list)), test_accuracy_list, label=f'{label_prefix} Test Accuracy', color='yellow', linestyle='--')

# third plot: valid loss across different lambda values
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, valid_loss_list in enumerate(all_valid_loss):
    label_prefix = f'extra_mp_{extra_layers[i]}'
    ax.plot(valid_loss_list, label=f'{label_prefix} valid loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Valid Loss')
ax.set_yscale("log")
plt.legend()
plt.savefig('{}/{}_valid_loss_with_different_extra_mp_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot
plt.close(fig)

# fourth plot: test loss across different lambda values
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, test_loss_list in enumerate(all_test_loss):
    label_prefix = f'{label}_{extra_layers[i]}'
    ax.plot(test_loss_list, label=f'{label_prefix} valid loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
ax.set_yscale("log")
plt.legend()
plt.savefig('{}/{}_test_loss_with_different_extra_mp_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot
plt.close(fig)


# fifth plot: test accuracy across different lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, test_accuracy_list in enumerate(all_test_acc):
    label_prefix = f'{label}_{extra_layers[i]}'
    ax.plot(test_accuracy_list, label=f'{label_prefix} test accuracy', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.legend()
plt.savefig('{}/{}_test_accuracy_with_different_extra_mp_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot
plt.close(fig)

# sixth plot: train accuracy across different lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, train_accuracy_list in enumerate(all_train_acc):
    label_prefix = f'{label}_{extra_layers[i]}'
    ax.plot(train_accuracy_list, label=f'{label_prefix} train accuracy', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.legend()
plt.savefig('{}/{}_train_accuracy_with_different_extra_mp_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot
plt.close(fig)

# seventh plot: valid accuracy across different lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, valid_accuracy_list in enumerate(all_valid_acc):
    label_prefix = f'{label}_{extra_layers[i]}'
    ax.plot(valid_accuracy_list, label=f'{label_prefix} valid accuracy', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Valid Accuracy')
plt.legend()
plt.savefig('{}/{}_valid_accuracy_with_different_extra_mp_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot
plt.close(fig)

# eighth plot: best test, best val across diffferent lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
ax.plot(range(len(extra_layers)), best_vals, label=f'Best valid accuracy', linestyle='-', marker='o')
ax.plot(range(len(extra_layers)), best_tests, label=f'Best test accuracy', linestyle='-', marker='o')

ax.set_xticks(range(len(extra_layers)))
ax.set_xticklabels(extra_layers)

plt.xlabel('Extra MP')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('{}/{}_best_accuracy_with_different_extra_mp_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot
plt.close(fig)

# nineth plot: train learning loss across different lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, train_learning_loss_list in enumerate(all_train_learning_loss):
    label_prefix = f'{label}_{extra_layers[i]}'
    ax.plot(train_learning_loss_list, label=f'{label_prefix} train learning loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Train Learning Loss')
ax.set_yscale("log")
plt.legend()
plt.savefig('{}/{}_train_learning_loss_with_different_extra_mp_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot
plt.close(fig)

# tenth plot: train learning loss across different lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, valid_learning_loss_list in enumerate(all_valid_learning_loss):
    label_prefix = f'{label}_{extra_layers[i]}'
    ax.plot(valid_learning_loss_list, label=f'{label_prefix} valid learning loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Valid Learning Loss')
ax.set_yscale("log")
plt.legend()
plt.savefig('{}/{}_valid_learning_loss_with_different_extra_mp_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot
plt.close(fig)

# nineth plot: train learning loss across different lambda
fig, ax = add_hyperparameter_text(params)
# Plot with evenly spaced points
for i, test_learning_loss_list in enumerate(all_test_learning_loss):
    label_prefix = f'{label}_{extra_layers[i]}'
    ax.plot(test_learning_loss_list, label=f'{label_prefix} test learning loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Test Learning Loss')
ax.set_yscale("log")
plt.legend()
plt.savefig('{}/{}_test_learning_loss_with_different_extra_mp_{}.png'.format(folder_name, dataset_name, get_timestamp()))
plt.clf()  # Clear the current figure for the next plot
plt.close(fig)
