import torch
# from torch_geometric.loader import DataLoader
# from torch_geometric.utils import degree
# from ogb.graphproppred import PygGraphPropPredDataset
# from ogb.graphproppred import Evaluator
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
    loss = criterion(out[train_idx], data.y[train_idx])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, criterion, data, train_split_idx, validation_split_idx, test_split_idx):
    model.eval()
    out = model(data)
    # train_loss = criterion(out[train_split_idx], data.y[train_split_idx])
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

    return train_acc, valid_acc, test_acc, valid_loss, test_loss


def run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim, fl_hidden_dim,
        epsilon, optimizer_lr, loss_func, total_epoch, index, freeze):
    ###############################
    # hardcoded value goes here!!!!
    # dataset_name = 'Cora'
    # num_mp_layers = 4
    # num_fl_layers = 2 # number of mlp layer
    # mp_hidden_dim = 8000
    # fl_hidden_dim = 512
    # epsilon = 5**0.5/2
    # optimizer_lr = 0.01
    # # weight_decay=5e-4
    # loss_func = 'CrossEntropyLoss'
    # total_epoch = 300
    display_step = 50
    # dropout=0
    # warm_up_epoch_num = 0
    # first_layer_linear=False
    # batch_normalization = False
    # skip_connection = False
    ###############################
    # fix_seed()
    dataset = Planetoid(root='data/Planetoid', name=dataset_name, transform=T.NormalizeFeatures())
    # print(f'Dataset: {dataset}:')
    # print('Number of graphs:', len(dataset))
    # print('Number of features:', dataset.num_features)
    # print('Number of classes:', dataset.num_classes)

    data = dataset[0]  # Cora has only one graph

    # print(data)
    # print('Number of nodes:', data.num_nodes)
    # print('Number of edges:', data.num_edges)
    # print('Training nodes:', data.train_mask.sum().item())

    d = data.x.shape[1]
    c = max(data.y.max().item() + 1, data.y.shape[0])

    # data split for train, val, and test
    train_idx = torch.where(data.train_mask)[0]
    valid_idx = torch.where(data.val_mask)[0]
    test_idx = torch.where(data.test_mask)[0]


    # Enable to find number of distinct neighborhood structures if needed
    k, labels = wl_relabel(data, 30)
    print(f'num distinct structures: {k}')
    # Evaluate OOD: check train, test distinct structures distribution
    train_k, test_k, train_test_overlap_k = wl_train_test_ood(labels, train_idx, test_idx)
    print(f'num distinct structures in training data: {train_k}, number of distinct structures in test data: {test_k}')
    print(f'num distinct structures exists in both training data and test data: {train_test_overlap_k}')

    # random search
    # initial_num_distinct_features = torch.unique(data.x, dim=0).float().size(0)
    # print('initial_num_distinct_features: ', initial_num_distinct_features)
    # rank_of_distinct_matrix = torch.linalg.matrix_rank(torch.unique(data.x, dim=0), tol=1e-5)
    # print('initial rank of distinct matrix: ', rank_of_distinct_matrix.item())

    model = DecoupleModel (
        # edge_index=data.edge_index,
        in_dim=d,
        out_dim=c,
        mp_width=mp_hidden_dim,
        fl_width=fl_hidden_dim,
        num_mp_layers = num_mp_layers,
        num_fl_layers = num_fl_layers,
        eps=epsilon,
        freeze=freeze
        # dropout=dropout,
        # first_layer_linear=first_layer_linear,
        # batch_normalization=batch_normalization,
        # skip_connection=skip_connection
    )
    # summary(model, input_data=(data))

    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)
    criterion = torch.nn.CrossEntropyLoss()
    if loss_func == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_func == 'NLLLoss':
        criterion = torch.nn.NLLLoss()
    # model.reset_parameters()

    print('Experiment run {}'.format(index))
    print(f'dataset: {dataset_name}')
    print(f'num_mp_layers: {num_mp_layers}')
    print(f'num_fl_layers: {num_fl_layers}')
    print(f'mp_hidden_dim: {mp_hidden_dim}')
    print(f'fl_hidden_dim: {fl_hidden_dim}')
    print(f'epsilon: {epsilon}')
    print(f'optimizer_lr: {optimizer_lr}')
    print(f'loss_func: {loss_func}')
    print(f'total_epoch: {total_epoch}')

    with open('experiment_records.txt', 'a') as f:
        f.writelines('Experiment run: \n')
        f.writelines('dataset: ' + dataset_name + '\n')
        f.writelines('num_mp_layers: ' + str(num_mp_layers) + '\n')
        f.writelines('num_fl_layers: ' + str(num_fl_layers) + '\n')
        f.writelines('mp_hidden_dim: ' + str(mp_hidden_dim) + '\n')
        f.writelines('fl_hidden_dim: ' + str(fl_hidden_dim) + '\n')
        f.writelines('epsilon: ' + str(epsilon) + '\n')
        f.writelines('optimizer_lr: ' + str(optimizer_lr) + '\n')
        # f.writelines('weight_decay: ' + str(weight_decay) + '\n')
        f.writelines('loss_func: ' + loss_func + '\n')
        f.writelines('total_epoch: ' + str(total_epoch) + '\n')
        # f.writelines('warm_up_epoch_num: ' + str(warm_up_epoch_num) + '\n')
        # f.writelines('first_layer_linear: ' + str(first_layer_linear) + '\n')
        # f.writelines('batch_normalization: ' + str(batch_normalization) + '\n')
        # f.writelines('skip_connection: ' + str(skip_connection) + '\n')
        # f.writelines('dropout: ' + str(dropout) + '\n')

    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []
    best_val = float('-inf')
    best_test = float('-inf')
    train_accuracy_list = []
    valid_accuracy_list = []
    test_accuracy_list = []

    for epoch in tqdm(range(1,total_epoch)):
        # if epoch < warm_up_epoch_num:
        #     model.turn_on_training()
        # else:
        #     model.turn_off_training()
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
        'dataset_name': 'Cora',
        'num_mp_layers': num_mp_layers,
        'num_fl_layers': num_fl_layers,
        'mp_hidden_dim': mp_hidden_dim,
        'fl_hidden_dim': fl_hidden_dim,
        'epsilon': epsilon,
        'optimizer_lr': optimizer_lr,
        'freeze': freeze
    }
    fig, ax = add_hyperparameter_text(params)
    # plt.figure(figsize=(10, 5))
    ax.plot(train_loss_list, label='Train Loss', color='blue')
    ax.plot(valid_loss_list, label='Valid Loss', color='orange')
    ax.plot(test_loss_list, label='Test Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.savefig('{}/loss_cora_{}_{}.png'.format(folder_name, index, timestamp))
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
    plt.savefig('{}/accuracy_cora{}_{}.png'.format(folder_name, index, timestamp))
    # plt.clf()  # Clear the current figure for the next plot
    plt.close()

    return best_val, best_test

def ablation_study_on_mp_depth(freeze):
    ###############################
    # Experiment setup
    dataset_name = 'Cora'
    num_mp_layers = 4
    num_fl_layers = 2 # number of mlp layer
    mp_hidden_dim = 3000
    fl_hidden_dim = 512
    epsilon = 5**0.5/2
    optimizer_lr = 0.01
    # weight_decay=5e-4
    loss_func = 'CrossEntropyLoss'
    total_epoch = 400
    ###############################
    print("Begin abalation study")
    index = 0
    # best_vals = []
    # best_tests = []
    candidates = [1,2,3,4,5,6,7]
    best_valid_accuracy_runs = np.zeros((num_runs, len(candidates)))
    best_test_accuracy_runs = np.zeros((num_runs, len(candidates)))
    for j, num_mp_layers in enumerate(candidates):
        for i in range(num_runs):
            best_val, best_test = run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim,
                                      fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index, freeze)
            best_valid_accuracy_runs[i][j] = best_val
            best_test_accuracy_runs[i][j] = best_test
        # best_vals.append(best_val)
        # best_tests.append(best_test)
        index += 1

    # Compute mean and std
    val_mean_acc = np.mean(best_valid_accuracy_runs, axis=0)
    test_mean_acc = np.mean(best_test_accuracy_runs, axis=0)
    val_std_acc = np.std(best_valid_accuracy_runs, axis=0)
    test_std_acc = np.std(best_test_accuracy_runs, axis=0)

    params = {
        'dataset_name': 'Cora',
        'num_fl_layers': 2,
        'mp_hidden_dim': 3000,
        'fl_hidden_dim' : 512,
        'epsilon': 5**0.5/2,
        'optimizer_lr': 0.01,
        'freeze': freeze
    }
    fig, ax = add_hyperparameter_text(params)
    # ax.plot(candidates, best_vals, label='Best Valid Accuracy', color='blue')
    # ax.plot(candidates, best_tests, label='Best Test Accuracy', color='red')

    # Plot valid
    ax.plot(candidates, val_mean_acc, label='Best Valid Accuracy', color='blue')              # Solid mean
    ax.plot(candidates, val_mean_acc + val_std_acc, linestyle='--', color='blue', alpha=0.3)  # Mean + std
    ax.plot(candidates, val_mean_acc - val_std_acc, linestyle='--', color='blue', alpha=0.3)  # Mean - std

    # Plot test
    ax.plot(candidates, test_mean_acc, label='Best Test Accuracy', color='red')                # Solid mean
    ax.plot(candidates, test_mean_acc + test_std_acc, linestyle='--', color='red', alpha=0.3)  # Mean + std
    ax.plot(candidates, test_mean_acc - test_std_acc, linestyle='--', color='red', alpha=0.3)  # Mean - std

    plt.xlabel('mp depth')
    plt.ylabel('Accuracy')
    plt.title('accuracy vs mp depth')
    plt.legend()
    plt.savefig('{}/mp_depth_accuracy{}_{}.png'.format(folder_name, index, timestamp))
    plt.clf()  # Clear the current figure for the next plot
    print("End abalation study")

def ablation_study_on_mp_width(freeze):
    ###############################
    # Experiment setup
    dataset_name = 'Cora'
    num_mp_layers = 3
    num_fl_layers = 2 # number of mlp layer
    mp_hidden_dim = 8000
    fl_hidden_dim = 512
    epsilon = 5**0.5/2
    optimizer_lr = 0.01
    # weight_decay=5e-4
    loss_func = 'CrossEntropyLoss'
    total_epoch = 400
    ###############################
    print("Begin abalation study")
    index = 0
    # best_vals = []
    # best_tests = []
    candidates = [250,500,750,1000,2000,4000,8000,16000]
    best_valid_accuracy_runs = np.zeros((num_runs, len(candidates)))
    best_test_accuracy_runs = np.zeros((num_runs, len(candidates)))
    for j, mp_hidden_dim in enumerate(candidates):
        for i in range(num_runs):
            best_val, best_test = run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim,
                                      fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index, freeze)
            best_valid_accuracy_runs[i][j] = best_val
            best_test_accuracy_runs[i][j] = best_test
        # best_vals.append(best_val)
        # best_tests.append(best_test)
        index += 1

    # Compute mean and std
    val_mean_acc = np.mean(best_valid_accuracy_runs, axis=0)
    test_mean_acc = np.mean(best_test_accuracy_runs, axis=0)
    val_std_acc = np.std(best_valid_accuracy_runs, axis=0)
    test_std_acc = np.std(best_test_accuracy_runs, axis=0)

    params = {
        'dataset_name': 'Cora',
        'num_mp_layers': 3,
        'num_fl_layers': 2,
        'fl_hidden_dim' : 512,
        'epsilon': 5**0.5/2,
        'optimizer_lr': 0.01,
        'freeze': freeze
    }
    fig, ax = add_hyperparameter_text(params)
    # ax.plot(candidates, best_vals, label='Best Valid Accuracy', color='blue')
    # ax.plot(candidates, best_tests, label='Best Test Accuracy', color='red')

    # Plot valid
    ax.plot(candidates, val_mean_acc, label='Best Valid Accuracy', color='blue')              # Solid mean
    ax.plot(candidates, val_mean_acc + val_std_acc, linestyle='--', color='blue', alpha=0.3)  # Mean + std
    ax.plot(candidates, val_mean_acc - val_std_acc, linestyle='--', color='blue', alpha=0.3)  # Mean - std

    # Plot test
    ax.plot(candidates, test_mean_acc, label='Best Test Accuracy', color='red')                # Solid mean
    ax.plot(candidates, test_mean_acc + test_std_acc, linestyle='--', color='red', alpha=0.3)  # Mean + std
    ax.plot(candidates, test_mean_acc - test_std_acc, linestyle='--', color='red', alpha=0.3)  # Mean - std

    plt.xlabel('mp width')
    plt.ylabel('Accuracy')
    plt.title('accuracy vs mp width')
    plt.legend()
    plt.savefig('{}/mp_width_accuracy{}_{}.png'.format(folder_name, index, timestamp))
    plt.clf()  # Clear the current figure for the next plot
    print("End abalation study")


def ablation_study_on_fc_depth(freeze):
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
    ###############################
    print("Begin abalation study")
    index = 0
    best_vals = []
    best_tests = []
    candidates = [1,2,3,4,5,6]
    best_valid_accuracy_runs = np.zeros((num_runs, len(candidates)))
    best_test_accuracy_runs = np.zeros((num_runs, len(candidates)))
    for j, num_fl_layers in enumerate(candidates):
        for i in range(num_runs):
            best_val, best_test = run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim,
                                      fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index, freeze)
            best_valid_accuracy_runs[i][j] = best_val
            best_test_accuracy_runs[i][j] = best_test
        # best_vals.append(best_val)
        # best_tests.append(best_test)
        index += 1

    # Compute mean and std
    val_mean_acc = np.mean(best_valid_accuracy_runs, axis=0)
    test_mean_acc = np.mean(best_test_accuracy_runs, axis=0)
    val_std_acc = np.std(best_valid_accuracy_runs, axis=0)
    test_std_acc = np.std(best_test_accuracy_runs, axis=0)

    params = {
        'dataset_name': 'Cora',
        'num_mp_layers': 3,
        'mp_hidden_dim': 3000,
        'fl_hidden_dim': 512,
        'epsilon': 5**0.5/2,
        'optimizer_lr': 0.01,
        'freeze': freeze
    }
    fig, ax = add_hyperparameter_text(params)
    # ax.plot(candidates, best_vals, label='Best Valid Accuracy', color='blue')
    # ax.plot(candidates, best_tests, label='Best Test Accuracy', color='red')

    # Plot valid
    ax.plot(candidates, val_mean_acc, label='Best Valid Accuracy', color='blue')              # Solid mean
    ax.plot(candidates, val_mean_acc + val_std_acc, linestyle='--', color='blue', alpha=0.3)  # Mean + std
    ax.plot(candidates, val_mean_acc - val_std_acc, linestyle='--', color='blue', alpha=0.3)  # Mean - std

    # Plot test
    ax.plot(candidates, test_mean_acc, label='Best Test Accuracy', color='red')                # Solid mean
    ax.plot(candidates, test_mean_acc + test_std_acc, linestyle='--', color='red', alpha=0.3)  # Mean + std
    ax.plot(candidates, test_mean_acc - test_std_acc, linestyle='--', color='red', alpha=0.3)  # Mean - std

    plt.xlabel('fc depth')
    plt.ylabel('Accuracy')
    plt.title('accuracy vs fc depth')
    plt.legend()
    plt.savefig('{}/fc_depth_accuracy{}_{}.png'.format(folder_name, index, timestamp))
    plt.clf()  # Clear the current figure for the next plot
    print("End abalation study")


def ablation_study_on_fc_width(freeze):
    ###############################
    # Experiment setup
    dataset_name = 'Cora'
    num_mp_layers = 3
    num_fl_layers = 2 # number of mlp layer
    mp_hidden_dim = 3000
    # fl_hidden_dim = 512
    epsilon = 5**0.5/2
    optimizer_lr = 0.01
    # weight_decay=5e-4
    loss_func = 'CrossEntropyLoss'
    total_epoch = 400
    ###############################
    print("Begin abalation study")
    index = 0
    # best_vals = []
    # best_tests = []
    candidates = [16,32,64,128,256,512,1024,2048,4096]
    best_valid_accuracy_runs = np.zeros((num_runs, len(candidates)))
    best_test_accuracy_runs = np.zeros((num_runs, len(candidates)))
    for j, fl_hidden_dim in enumerate(candidates):
        for i in range(num_runs):
            best_val, best_test = run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim,
                                      fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index, freeze)
            best_valid_accuracy_runs[i][j] = best_val
            best_test_accuracy_runs[i][j] = best_test
        # best_vals.append(best_val)
        # best_tests.append(best_test)
        index += 1

    # Compute mean and std
    val_mean_acc = np.mean(best_valid_accuracy_runs, axis=0)
    test_mean_acc = np.mean(best_test_accuracy_runs, axis=0)
    val_std_acc = np.std(best_valid_accuracy_runs, axis=0)
    test_std_acc = np.std(best_test_accuracy_runs, axis=0)

    params = {
        'dataset_name': 'Cora',
        'num_mp_layers': 3,
        'num_fl_layers': 2,
        'mp_hidden_dim': 3000,
        'epsilon': 5**0.5/2,
        'optimizer_lr': 0.01,
        'freeze': freeze
    }
    fig, ax = add_hyperparameter_text(params)
    # plt.figure(figsize=(10, 5))
    # ax.plot(candidates, best_vals, label='Best Valid Accuracy', color='blue')
    # ax.plot(candidates, best_tests, label='Best Test Accuracy', color='red')

    # Plot valid
    ax.plot(candidates, val_mean_acc, label='Best Valid Accuracy', color='blue')              # Solid mean
    ax.plot(candidates, val_mean_acc + val_std_acc, linestyle='--', color='blue', alpha=0.3)  # Mean + std
    ax.plot(candidates, val_mean_acc - val_std_acc, linestyle='--', color='blue', alpha=0.3)  # Mean - std

    # Plot test
    ax.plot(candidates, test_mean_acc, label='Best Test Accuracy', color='red')                # Solid mean
    ax.plot(candidates, test_mean_acc + test_std_acc, linestyle='--', color='red', alpha=0.3)  # Mean + std
    ax.plot(candidates, test_mean_acc - test_std_acc, linestyle='--', color='red', alpha=0.3)  # Mean - std

    plt.xlabel('fc width')
    plt.ylabel('Accuracy')
    plt.title('accuracy vs fc width')
    plt.legend()
    plt.savefig('{}/fc_width_accuracy{}_{}.png'.format(folder_name, index, timestamp))
    plt.clf()  # Clear the current figure for the next plot
    print("End abalation study")


def ablation_study(freeze):
    ###############################
    # Experiment setup
    dataset_name = 'Cora'
    num_mp_layers = 4
    num_fl_layers = 2 # number of mlp layer
    mp_hidden_dim = 8000
    fl_hidden_dim = 512
    epsilon = 5**0.5/2
    optimizer_lr = 0.01
    # weight_decay=5e-4
    loss_func = 'CrossEntropyLoss'
    total_epoch = 1000
    ###############################
    print("Begin abalation study")
    index = 0
    for num_mp_layers in [1,2,3,4,5,6]:
        for num_fl_layers in [1,2,3,4,5]:
            for mp_hidden_dim in [16,32,64,128,256,512,1024,2048,4012,8024]:
                for fl_hidden_dim in [16,32,64,128,256,512,1024,2048]:
                    run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim,
                        fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index, freeze)
                    index += 1
    print("End abalation study")




parser = argparse.ArgumentParser(description="Process experiment arguments")
parser.add_argument('--mp_depth', action='store_true', help='message passing layer depth')
parser.add_argument('--mp_width', action='store_true', help='message passing layer width')
parser.add_argument('--fc_depth', action='store_true', help='fully connected layer depth')
parser.add_argument('--fc_width', action='store_true', help='fully connected layer width')
parser.add_argument('--train_mp', action='store_true', help='train message passing layers')
parser.add_argument('--num_runs', type=int, default=1, help='num of runs per setting.')

args = parser.parse_args()
freeze = not args.train_mp

# For filename
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# Create folder for results
folder = Path(f"result_{timestamp}")
folder.mkdir(parents=True, exist_ok=True)
folder_name = folder.name

# print the whole command line arguments
print(sys.executable, ' '.join(sys.argv))

num_runs = args.num_runs if args.num_runs > 0 else 1
print('num_runs {}'.format(num_runs))

if args.mp_depth:
    ablation_study_on_mp_depth(freeze)
if args.mp_width:
    ablation_study_on_mp_width(freeze)
if args.fc_depth:
    ablation_study_on_fc_depth(freeze)
if args.fc_width:
    ablation_study_on_fc_width(freeze)


# ablation_study()

# dataset_name = 'Cora'
# num_mp_layers = 4
# num_fl_layers = 2 # number of mlp layer
# mp_hidden_dim = 4000
# fl_hidden_dim = 512
# epsilon = 5**0.5/2
# optimizer_lr = 0.01
# # weight_decay=5e-4
# loss_func = 'CrossEntropyLoss'
# total_epoch = 300
# ###############################
# index = 10000
# run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim,
#     fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index)

