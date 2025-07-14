import torch
import numpy as np

# TODO: deal with multi-splits train_mask cases

def rand_train_test_idx(label, train_prop=0.5, valid_prop=0.25):
    """randomly splits label into train/valid/test splits"""
    labeled_data_idx = torch.where(label != -1)[0]
    n = labeled_data_idx.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.randperm(n)  # random indices
    labeled_data_idx = labeled_data_idx[perm]

    train_idx = labeled_data_idx[:train_num]
    valid_idx = labeled_data_idx[train_num : train_num + valid_num]
    test_idx = labeled_data_idx[train_num + valid_num :]

    # train_idx = label[train_indices]
    # valid_idx = label[val_indices]
    # test_idx = label[test_indices]

    return train_idx, valid_idx, test_idx
