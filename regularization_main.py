from main import run_overfitting_understanding
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




def overfit_experiment():
    ###############################
    # Experiment setup
    dataset_name = 'Cora'
    num_mp_layers = 3
    num_fl_layers = 2 # number of mlp layer
    mp_hidden_dim = 3000
    fl_hidden_dim = 512
    epsilon = 5**0.5/2
    optimizer_lr = 0.01
    # weight_decay=5e-4
    loss_func = 'CrossEntropyLoss'
    total_epoch = 400
    freeze=False
    ###############################
    extra_data_rate = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    best_vals = np.zeros(len(extra_data_rate))
    best_tests = np.zeros(len(extra_data_rate))
    for i, rate in enumerate(extra_data_rate):
        best_val, best_test, _ = run_overfitting_understanding(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim,
                                                                fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index=0,
                                                                freeze=freeze, extra_train_data_rate=rate)
        best_vals[i] = best_val
        best_tests[i] = best_test

    params = {
        'dataset_name': dataset_name,
        'num_fl_layers': num_fl_layers,
        'mp_hidden_dim': mp_hidden_dim,
        'fl_hidden_dim' : fl_hidden_dim,
        'epsilon': epsilon,
        'optimizer_lr': optimizer_lr,
        'freeze': freeze
    }
    fig, ax = add_hyperparameter_text(params)
    ax.plot(extra_data_rate, best_vals, label='Best Valid Accuracy', color='blue')
    ax.plot(extra_data_rate, best_tests, label='Best Test Accuracy', color='red')

    plt.xlabel('Extra train data rate')
    plt.ylabel('Accuracy')
    plt.title('accuracy vs Extra train data')
    plt.legend()
    plt.savefig('{}/{}_overfit_experiment_accuracy_{}.png'.format(folder_name, dataset_name, timestamp))
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
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# Create folder for results
folder = Path(f"result_{timestamp}")
folder.mkdir(parents=True, exist_ok=True)
folder_name = folder.name

# print the whole command line arguments
# print(sys.executable, ' '.join(sys.argv))

# num_runs = args.num_runs if args.num_runs > 0 else 1
# print('num_runs {}'.format(num_runs))

overfit_experiment()

