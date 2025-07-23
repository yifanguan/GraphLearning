from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import LINKXDataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import WikiCS
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Reddit
from torch_geometric.data import Data
import torch_geometric.transforms as T

# from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
# NodePropPredDataset, 
from ogb.nodeproppred import PygNodePropPredDataset

# import dgl.function as fn

import numpy as np
import torch
import os
import gdown
import scipy

def load_dataset(data_dir, dataset_name, filter=None):
    """ Loader for Dataset
        Returns Dataset
    """
    data = None
    dataset_name = dataset_name.lower()
    if dataset_name in ('texas'):
        data = load_webkb_dataset(data_dir, dataset_name)
    elif dataset_name in ('cora', 'citeseer', 'pubmed'):
        data = load_planetoid_dataset(data_dir, dataset_name)
    elif dataset_name in ('cornell5'):
        data = load_facebook_100_dataset(data_dir, dataset_name)
    elif dataset_name in ('mnist'):
        data = load_mnist_dataset(data_dir, dataset_name, filter=filter)
    elif dataset_name in  ('amazon-photo', 'amazon-computers'):
        data = load_amazon_dataset(data_dir, dataset_name)
    elif dataset_name in  ('coauthor-cs', 'coauthor-physics'):
        data = load_coauthor_dataset(data_dir, dataset_name)
    elif dataset_name in ('wikics'):
        data = load_wikics_dataset(data_dir, dataset_name)
    elif dataset_name in ('chameleon', 'squirrel'):
        data = load_wiki_new(data_dir, dataset_name)
    elif dataset_name in ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'):
        data = load_hetero_dataset(data_dir, dataset_name)
    elif dataset_name in ('ogbn-arxiv', 'ogbn-products'):
        data = load_ogb_dataset(data_dir, dataset_name)
    elif dataset_name == 'pokec':
        data = load_pokec_mat(data_dir)
    elif dataset_name in ('ogbn-proteins'):
        data = load_ogb_proteins_dataset(data_dir, dataset_name)
    elif dataset_name == 'reddit-binary':
        data = load_reddit_binary_dataset(data_dir, dataset_name, filter)
    elif dataset_name == 'reddit':
        data = load_reddit_dataset(data_dir, dataset_name)
    else:
        raise ValueError('Invalid dataname')
    return data


def load_ogb_proteins_dataset(data_dir, dataset_name):
    '''
    Proteins dataset doesn't have node features, requring preprocessing.
    Data(num_nodes=132534, edge_index=[2, 79122504], edge_attr=[79122504, 8], node_species=[132534, 1], y=[132534, 112])

    https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/proteins/gnn.py
    https://arxiv.org/pdf/2005.00687
    '''
    n_node_feats, n_edge_feats, n_classes = 0, 8, 112
    data = DglNodePropPredDataset(root=f'{data_dir}/OGBN', name=dataset_name)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    graph.ndata["labels"] = labels

    # The sum of the weights of adjacent edges is used as node features.
    graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))
    n_node_feats = graph.ndata["feat"].shape[-1]

    # Only the labels in the training set are used as features, while others are filled with zeros.
    graph.ndata["train_labels_onehot"] = torch.zeros(graph.number_of_nodes(), n_classes)
    graph.ndata["train_labels_onehot"][train_idx, labels[train_idx, 0]] = 1
    graph.ndata["deg"] = graph.out_degrees().float().clamp(min=1)

    graph.create_formats_()

    # Convert to PyG graph
    src, dst = graph.edges()
    edge_index = torch.stack([src, dst], dim=0)  # [2, num_edges]

    # Example: node features
    x = graph.ndata['feat'] if 'feat' in graph.ndata else None

    # Example: edge features
    # edge_attr = graph.edata['feat'] if 'feat' in graph.edata else None

    # Example: labels
    y = graph.ndata['labels'] if 'labels' in graph.ndata else None

    data = Data(x=x, edge_index=edge_index, y=y)

    return data
    # return graph, labels



def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    if not os.path.exists(f'{data_dir}/Pokec/pokec.mat'):
        if not os.path.exists(f'{data_dir}/Pokec'):
            os.mkdir(f'{data_dir}/Pokec')
        drive_id = '1575QYJwJlj7AWuOKMlwVmMz8FcslUncu'
        gdown.download(id=drive_id, output="data/Pokec/")
        #import sys; sys.exit()
        #gdd.download_file_from_google_drive(
        #    file_id= drive_id, \
        #    dest_path=f'{data_dir}/pokec/pokec.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    labels = torch.tensor(fulldata['label'].flatten(), dtype=torch.long)

    data = Data(x=node_feat, edge_index=edge_index, y=labels)

    # data = T.ToSparseTensor()(data)

    return data

def load_reddit_dataset(data_dir, name):
    # This will download and process the Reddit dataset into 'data/Reddit'
    dataset = Reddit(root=f'{data_dir}/Reddit')
    data = dataset[0]
    return data


def load_reddit_binary_dataset(data_dir, name, filter):
    dataset = TUDataset(root=f'{data_dir}/Reddit', name=name)
    if filter == 'all':
        return dataset

    data = dataset[0]

    return data


def load_ogb_dataset(data_dir, name):
    '''
    Different from classic GNN, which does raw processing
    '''
    # Load dataset (with PyG format)
    # dataset = PygNodePropPredDataset(root=f'{data_dir}/OGBN', name=name, transform=T.ToSparseTensor())
    dataset = PygNodePropPredDataset(root=f'{data_dir}/OGBN', name=name)
    data = dataset[0]

    # Split indices
    # split_idx = dataset.get_idx_split()
    # train_idx = split_idx['train']
    # val_idx = split_idx['valid']
    # test_idx = split_idx['test']

    return data


def load_hetero_dataset(data_dir, name, norm_feature=False):
    if norm_feature:
        transform = T.NormalizeFeatures()
        dataset = HeterophilousGraphDataset(root=f'{data_dir}/Hetero',
                                            name=name.capitalize(), transform=transform)
    else:
        dataset = HeterophilousGraphDataset(root=f'{data_dir}/Hetero',
                                            name=name.capitalize())
    data = dataset[0]

    return data


def load_wiki_new(data_dir, name, norm_feature=False):
    path = f'{data_dir}/geom-gcn/{name}/{name}_filtered.npz'
    data = np.load(path)
    # in case we want to check what is stored in the npzfile
    # lst = data.files
    # for item in lst:
    #     print(item)
    node_feat = data['node_features'] # unnormalized
    node_feat=torch.as_tensor(node_feat)

    labels = data['node_labels']
    labels=torch.as_tensor(labels)

    edges = data['edges'] # dim: (E, 2)
    edge_index = edges.T
    edge_index=torch.as_tensor(edge_index)

    data = Data(x=node_feat, edge_index=edge_index, y=labels)

    if norm_feature:
        transform = T.NormalizeFeatures()
        data = transform(data)

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

def load_amazon_dataset(data_dir, name, norm_feature=True):
    if name == 'amazon-photo':
        if norm_feature:
            transform = T.NormalizeFeatures()
            dataset = Amazon(root=f'{data_dir}/Amazon',
                                   name='Photo', transform=transform)
        else:
            dataset = Amazon(root=f'{data_dir}/Amazon',
                                   name='Photo')
    elif name == 'amazon-computers':
        if norm_feature:
            transform = T.NormalizeFeatures()
            dataset = Amazon(root=f'{data_dir}/Amazon',
                                   name='Computers', transform=transform)
        else:
            dataset = Amazon(root=f'{data_dir}/Amazon',
                                   name='Computers')

    data = dataset[0]

    return data

def load_coauthor_dataset(data_dir, name, norm_feature=True):
    if name == 'coauthor-cs':
        if norm_feature:
            transform = T.NormalizeFeatures()
            dataset = Coauthor(root=f'{data_dir}/Coauthor',
                                name='CS', transform=transform)
        else:
            dataset = Coauthor(root=f'{data_dir}/Coauthor',
                                name='CS')
    elif name == 'coauthor-physics':
        if norm_feature:
            transform = T.NormalizeFeatures()
            dataset = Coauthor(root=f'{data_dir}/Coauthor',
                                name='Physics', transform=transform)
        else:
            dataset = Coauthor(root=f'{data_dir}/Coauthor',
                                name='Physics')

    data = dataset[0]

    return data


def load_wikics_dataset(data_dir, name, norm_feature=True):
    dataset = WikiCS(root=f'{data_dir}/WikiCS')
    data = dataset[0]

    return data


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

# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir, 'pubmed')
