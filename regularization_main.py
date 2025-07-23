from main import run_with_regularization
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




def regularization_experiment(dataset_name, num_mp_layers=3, num_fl_layers=2,
                                mp_hidden_dim=3000, fl_hidden_dim=512, epsilon=5**0.5/2, optimizer_lr=0.01,
                                loss_func='CrossEntropyLoss', total_epoch=400,
                                freeze=False):
    ###############################
    # Experiment setup
    # dataset_name = 'Cora'
    # num_mp_layers = 3
    # num_fl_layers = 2 # number of mlp layer
    # mp_hidden_dim = 3000
    # fl_hidden_dim = 512
    # epsilon = 5**0.5/2
    # optimizer_lr = 0.01
    # # weight_decay=5e-4
    # loss_func = 'CrossEntropyLoss'
    # total_epoch = 400
    # freeze=False
    # ###############################
    weight_decays = [10, 1, 5e-1, 1e-1, 1e-2, 5e-2, 1e-3, 5e-4, 1e-4, 1e-5, 0]
    best_vals = np.zeros(len(weight_decays))
    best_tests = np.zeros(len(weight_decays))
    for i, weight_decay in enumerate(weight_decays):
        best_val, best_test, _ = run_with_regularization(dataset_name, 'AdamW', weight_decay, num_mp_layers, num_fl_layers, mp_hidden_dim,
                                                         fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index=0,
                                                         freeze=freeze)
        best_vals[i] = best_val
        best_tests[i] = best_test

    params = {
        'dataset_name': dataset_name,
        'num_mp_layers': num_mp_layers,
        'num_fl_layers': num_fl_layers,
        'mp_hidden_dim': mp_hidden_dim,
        'fl_hidden_dim' : fl_hidden_dim,
        'epsilon': epsilon,
        'optimizer_lr': optimizer_lr,
        'freeze': freeze
    }
    fig, ax = add_hyperparameter_text(params)
    # Plot with evenly spaced points
    ax.plot(range(len(weight_decays)), best_vals, label='Best Valid Accuracy', color='blue', marker='o')
    ax.plot(range(len(weight_decays)), best_tests, label='Best Test Accuracy', color='red', marker='o')

    ax.set_xticks(range(len(weight_decays)))
    ax.set_xticklabels(weight_decays)

    plt.xlabel('Weight Decay')
    plt.ylabel('Accuracy')
    plt.title('accuracy vs weight decay')
    plt.legend()
    plt.savefig('{}/{}_adamw_weight_decay_experiment_accuracy_{}.png'.format(folder_name, dataset_name, timestamp))
    plt.clf()  # Clear the current figure for the next plot


def regularization_experiment_dropout(dataset_name, num_mp_layers=3, num_fl_layers=2,
                                        mp_hidden_dim=3000, fl_hidden_dim=512, epsilon=5**0.5/2, optimizer_lr=0.01,
                                        loss_func='CrossEntropyLoss', total_epoch=400,
                                        freeze=False, skip_connection=False, folder_name_suffix="", version='v2'):
    ###############################
    # Experiment setup
    # dataset_name = 'Cora'
    # num_mp_layers = 3
    # num_fl_layers = 2 # number of mlp layer
    # mp_hidden_dim = 3000
    # fl_hidden_dim = 512
    # epsilon = 5**0.5/2
    # optimizer_lr = 0.01
    weight_decay=0.01
    # loss_func = 'CrossEntropyLoss'
    # total_epoch = 400
    # freeze=False
    ###############################
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_vals = np.zeros(len(dropout_rates))
    best_tests = np.zeros(len(dropout_rates))
    for i, dropout in enumerate(dropout_rates):
        best_val, best_test, _ = run_with_regularization(dataset_name, 'AdamW', weight_decay, num_mp_layers, num_fl_layers, mp_hidden_dim,
                                                         fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index=0,
                                                         freeze=freeze, dropout=dropout, skip_connection=skip_connection,
                                                         folder_name_suffix=folder_name_suffix, version=version)
        best_vals[i] = best_val
        best_tests[i] = best_test

    params = {
        'dataset_name': dataset_name,
        'num_mp_layers': num_mp_layers,
        'num_fl_layers': num_fl_layers,
        'mp_hidden_dim': mp_hidden_dim,
        'fl_hidden_dim' : fl_hidden_dim,
        'optimizer_lr': optimizer_lr,
        'freeze': freeze,
        'skip_connection': skip_connection
    }
    fig, ax = add_hyperparameter_text(params)
    # Plot with evenly spaced points
    ax.plot(range(len(dropout_rates)), best_vals, label='Best Valid Accuracy', color='blue', marker='o')
    ax.plot(range(len(dropout_rates)), best_tests, label='Best Test Accuracy', color='red', marker='o')

    ax.set_xticks(range(len(dropout_rates)))
    ax.set_xticklabels(dropout_rates)

    plt.xlabel('Dropout Rate')
    plt.ylabel('Accuracy')
    plt.title('accuracy vs dropout rate')
    plt.legend()
    plt.savefig('{}/{}_adamw_dropout_experiment_accuracy_{}.png'.format(folder_name, dataset_name, get_timestamp()))
    plt.clf()  # Clear the current figure for the next plot


# parser = argparse.ArgumentParser(description="Process experiment arguments")
# parser.add_argument('--mp_depth', action='store_true', help='message passing layer depth')
# parser.add_argument('--mp_width', action='store_true', help='message passing layer width')
# parser.add_argument('--fc_depth', action='store_true', help='fully connected layer depth')
# parser.add_argument('--fc_width', action='store_true', help='fully connected layer width')
# parser.add_argument('--train_mp', action='store_true', help='train message passing layers')
# parser.add_argument('--num_runs', type=int, default=1, help='num of runs per setting.')

# args = parser.parse_args()
# freeze = not args.train_mp

# For filename
# now = datetime.now()
# timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
# timestamp = get_timestamp()

# Create folder for results
folder = Path(f"result_regularization")
folder.mkdir(parents=True, exist_ok=True)
folder_name = folder.name

# print the whole command line arguments
# print(sys.executable, ' '.join(sys.argv))

# num_runs = args.num_runs if args.num_runs > 0 else 1
# print('num_runs {}'.format(num_runs))

# regularization_experiment(dataset_name='citeseer', num_mp_layers=3, num_fl_layers=1,
#                             mp_hidden_dim=3000, fl_hidden_dim=256, epsilon=5**0.5/2,
#                             optimizer_lr=0.01,
#                             loss_func='CrossEntropyLoss', total_epoch=400,
#                             freeze=False)

# regularization_experiment(dataset_name='cora', num_mp_layers=3, num_fl_layers=1,
#                             mp_hidden_dim=3000, fl_hidden_dim=256, epsilon=5**0.5/2,
#                             optimizer_lr=0.01,
#                             loss_func='CrossEntropyLoss', total_epoch=400,
#                             freeze=False)


# generate_expressive_power_plot_with_training(dataset_name='Cora', mp_depth=3, tolerance=1e-5, skip_connection=False, dropout=0, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000],
#                                              num_fl_layers=5, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.001, total_epoch=500)


regularization_experiment_dropout(dataset_name='cora', num_mp_layers=3, num_fl_layers=3,
                            mp_hidden_dim=4000, fl_hidden_dim=4000, epsilon=5**0.5/2,
                            optimizer_lr=0.0001,
                            loss_func='CrossEntropyLoss', total_epoch=1000,
                            freeze=False, skip_connection=True, folder_name_suffix="v2_verify_dropout_cora_6_4000", version='v2')
# ---- Clean up GPU memory ----
torch.cuda.empty_cache()       # release cached blocks
gc.collect()                   # force Python to collect garbage
torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1


# regularization_experiment_dropout(dataset_name='cora', num_mp_layers=3, num_fl_layers=5,
#                             mp_hidden_dim=4000, fl_hidden_dim=128, epsilon=5**0.5/2,
#                             optimizer_lr=0.001,
#                             loss_func='CrossEntropyLoss', total_epoch=1000,
#                             freeze=False, skip_connection=False, folder_name_suffix="verify_dropout_cora_3")
# # ---- Clean up GPU memory ----
# torch.cuda.empty_cache()       # release cached blocks
# gc.collect()                   # force Python to collect garbage
# torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1



# # generate_expressive_power_plot_with_training(dataset_name='Cora', mp_depth=3, tolerance=1e-5, skip_connection=False, dropout=0, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000],
# #                                              num_fl_layers=5, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.001, total_epoch=500)

# regularization_experiment_dropout(dataset_name='citeseer', num_mp_layers=6, num_fl_layers=2,
#                             mp_hidden_dim=4000, fl_hidden_dim=128, epsilon=5**0.5/2,
#                             optimizer_lr=0.001,
#                             loss_func='CrossEntropyLoss', total_epoch=1000,
#                             freeze=False, skip_connection=False, folder_name_suffix="verify_dropout_citeseer_6")
# # ---- Clean up GPU memory ----
# torch.cuda.empty_cache()       # release cached blocks
# gc.collect()                   # force Python to collect garbage
# torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1
