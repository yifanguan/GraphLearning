from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import LINKXDataset
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T

def load_dataset(data_dir, dataset_name, filter=None):
    """ Loader for Dataset
        Returns Dataset
    """
    data = None
    dataset_name = dataset_name.lower()
    if dataset_name in ('texas'):
        data = load_webkb_dataset(data_dir, dataset_name)
    elif dataset_name in ('cora'):
        data = load_planetoid_dataset(data_dir, dataset_name)
    elif dataset_name in ('cornell5'):
        data = load_facebook_100_dataset(data_dir, dataset_name)
    elif dataset_name in ('mnist'):
        data = load_mnist_dataset(data_dir, dataset_name, filter=filter)
    else:
        raise ValueError('Invalid dataname')
    return data

def load_webkb_dataset(data_dir, name, norm_feature=True):
    if norm_feature:
        transform = T.NormalizeFeatures()
        dataset = WebKB(root=f'{data_dir}/WebKB',
                                  name=name, transform=transform)
    else:
        dataset = WebKB(root=f'{data_dir}/WebKB', name=name)
    data = dataset[0]

    return data


def load_planetoid_dataset(data_dir, name, norm_feature=True):
    if norm_feature:
        transform = T.NormalizeFeatures()
        dataset = Planetoid(root=f'{data_dir}/Planetoid',
                                  name=name, transform=transform)
    else:
        dataset = Planetoid(root=f'{data_dir}/Planetoid', name=name)
    data = dataset[0]

    return data


def load_facebook_100_dataset(data_dir, name, norm_feature=True):
    '''
    This load the preprocessed version LINKX-Cornell5: https://github.com/CUAI/Non-Homophily-Benchmarks
    There is also raw version in .mat format: socfb-Cornell5 
    '''
    if norm_feature:
        transform = T.NormalizeFeatures()
        dataset = LINKXDataset(root=f'{data_dir}/LINKX',
                                  name=name, transform=transform)
    else:
        dataset = LINKXDataset(root=f'{data_dir}/LINKX', name=name)
    data = dataset[0]

    return data

def load_mnist_dataset(data_dir, name, norm_feature=True, filter=None):
    '''
    MNIST Superpixels dataset contains multiple graphs

    filter: digit filter
    '''
    if norm_feature:
        transform = T.NormalizeFeatures()
        dataset = (
            MNISTSuperpixels(root=f'{data_dir}/MNISTSuperpixels', train=True, transform=transform) +
            MNISTSuperpixels(root=f'{data_dir}/MNISTSuperpixels', train=False, transform=transform)
        )
    else:
        dataset = (
            MNISTSuperpixels(root=f'{data_dir}/MNISTSuperpixels', train=True) +
            MNISTSuperpixels(root=f'{data_dir}/MNISTSuperpixels', train=False)
        )
    dataset = [data for data in dataset if data.y.item() == filter]

    return dataset

# class Dataset(InMemoryDataset):
#     '''
#     root: root directory where the dataset should be saved
#     dataset_name: name of the dataset
#     '''
#     def __init__(self, root, dataset_name, transform=None, pre_transform=None):
#         super().__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     # @property
#     # def processed_file_names(self):
#     #     return [f'{self.name}_data.pt']

#     def process(self):
#         # Load raw data depending on name
#         if self.name == 'cornell5':
#             data = self.load_cornell5()
#         elif self.name == 'texas':
#             data = self.load_texas()
#         else:
#             raise ValueError(f'Unknown dataset {self.name}')

#         data_list = [data]
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

#     def load_cornell5(self):
#         # Load from .mat, SNAP, or other format
#         # Then wrap as PyG Data object
#         return Data(x=..., edge_index=..., y=...)

#     def load_texas(self):
#         return Data(x=..., edge_index=..., y=...)
    
#     def load_cora(self):








# import torch
# import torch_geometric.transforms as T
# from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, WikiCS
# from ogb.nodeproppred import NodePropPredDataset

# import numpy as np
# import scipy.sparse as sp
# from os import path
# #from google_drive_downloader import GoogleDriveDownloader as gdd
# import gdown
# import scipy
# from torch_geometric.datasets import Planetoid
# from data_utils import dataset_drive_url, rand_train_test_idx


# class NCDataset(object):
#     def __init__(self, name):
#         """
#         based off of ogb NodePropPredDataset
#         https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
#         Gives torch tensors instead of numpy arrays
#             - name (str): name of the dataset
#             - root (str): root directory to store the dataset folder
#             - meta_dict: dictionary that stores all the meta-information about data. Default is None,
#                     but when something is passed, it uses its information. Useful for debugging for external contributers.

#         Usage after construction:

#         split_idx = dataset.get_idx_split()
#         train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
#         graph, label = dataset[0]

#         Where the graph is a dictionary of the following form:
#         dataset.graph = {'edge_index': edge_index,
#                          'edge_feat': None,
#                          'node_feat': node_feat,
#                          'num_nodes': num_nodes}
#         For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

#         """

#         self.name = name  # original name, e.g., ogbn-proteins
#         self.graph = {}
#         self.label = None

#     def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
#         """
#         train_prop: The proportion of dataset for train split. Between 0 and 1.
#         valid_prop: The proportion of dataset for validation split. Between 0 and 1.
#         """

#         if split_type == 'random':
#             ignore_negative = False if self.name == 'ogbn-proteins' else True
#             train_idx, valid_idx, test_idx = rand_train_test_idx(
#                 self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
#             split_idx = {'train': train_idx,
#                          'valid': valid_idx,
#                          'test': test_idx}

#         return split_idx

#     def __getitem__(self, idx):
#         assert idx == 0, 'This dataset has only one graph'
#         return self.graph, self.label

#     def __len__(self):
#         return 1

#     def __repr__(self):
#         return '{}({})'.format(self.__class__.__name__, len(self))



# def load_pokec_mat(data_dir):
#     """ requires pokec.mat """
#     if not path.exists(f'{data_dir}/pokec/pokec.mat'):
#         drive_id = '1575QYJwJlj7AWuOKMlwVmMz8FcslUncu'
#         gdown.download(id=drive_id, output="data/pokec/")
#         #import sys; sys.exit()
#         #gdd.download_file_from_google_drive(
#         #    file_id= drive_id, \
#         #    dest_path=f'{data_dir}/pokec/pokec.mat', showsize=True)

#     fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')

#     dataset = NCDataset('pokec')
#     edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
#     node_feat = torch.tensor(fulldata['node_feat']).float()
#     num_nodes = int(fulldata['num_nodes'])
#     dataset.graph = {'edge_index': edge_index,
#                      'edge_feat': None,
#                      'node_feat': node_feat,
#                      'num_nodes': num_nodes}

#     label = fulldata['label'].flatten()
#     dataset.label = torch.tensor(label, dtype=torch.long)
#     return dataset
