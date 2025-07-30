from main import run_with_regularization, run
from models.dln import DecoupleModel
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
import torch
import gc
import math


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
