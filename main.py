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
import math
import random
import numpy as np
from torchinfo import summary
from torch_geometric.data import Data
from utils.wl_test import wl_relabel



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

# result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

@torch.no_grad()
def evaluate(model, data, train_split_idx, validation_split_idx, test_split_idx):
    model.eval()
    out = model(data)
    # out = model(data.x, data.edge_index)
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

    return train_acc, valid_acc, test_acc 

def run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim, fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index):
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
    fix_seed()
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

    # Enable if needed
    # k = wl_relabel(data, 30)
    # print(f'num distinct structures: {k}')

    # 先算 distinct klog(k) distinct local structure
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
        # freeze
        # dropout=dropout,
        # first_layer_linear=first_layer_linear,
        # batch_normalization=batch_normalization,
        # skip_connection=skip_connection
    )
    # summary(model, input_data=(data))

    train_idx = torch.where(data.train_mask)[0]
    valid_idx = torch.where(data.val_mask)[0]
    test_idx = torch.where(data.test_mask)[0]

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

    loss_list = []
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
        loss_list.append(loss)
        train_acc, valid_acc, test_acc = evaluate(model, data, train_idx, valid_idx, test_idx)

        train_accuracy_list.append(train_acc)
        valid_accuracy_list.append(valid_acc)
        test_accuracy_list.append(test_acc)

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
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Loss', color='red')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.savefig('loss_cora{}.png'.format(index))
    plt.clf()  # Clear the current figure for the next plot
    # Plotting the acc in one figure
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy_list, label='Train Accuracy', color='blue')
    plt.plot(valid_accuracy_list, label='Valid Accuracy', color='orange')
    plt.plot(test_accuracy_list, label='Test Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.savefig('accuracy_cora{}.png'.format(index))
    plt.clf()  # Clear the current figure for the next plot


def ablation_study():
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
                        fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index)
                    index += 1
    print("End abalation study")



ablation_study()
