import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import sys, os
from datetime import datetime
import math
# Add parent folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dln import InjectiveMP, DecoupleModel, MPOnlyModel, iMP, iGNN
from utils.wl_test import wl_relabel, find_group
from utils.dataset import load_dataset
from utils.timestamp import get_timestamp
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Batch
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.nn import GIN
import os
import gc


# dataset_name = 'Cora'

# current_file = os.path.abspath(__file__)
# parent_dir = os.path.dirname(os.path.dirname(current_file))
# dataset = Planetoid(root=f'{parent_dir}/data/Planetoid', name=dataset_name, transform=T.NormalizeFeatures())
# data = dataset[0]  # Cora has only one graph

# dim_list = [50, 150, 300, 500, 1000, 2000, 4000, 8000]
# # linear_res = []
# non_linear_res = []
# labels = [f"Non_linear_mp/dim={dim}" for dim in dim_list]
# distinct_node_feature = []
# distinct_node_feature_x = []

# ###############
# tolerance=1e-7
# num_mp_layer=6
# ###############

# # 经过几轮message passing过后，每个node有distinct features。k distinct features
# # 最终number of distinct features不再增加 -》injective，学完了所有的structure。
# # m 升高dimension，提高linearly independent的可能性-》high rank
# # 我们最终希望rank = distinct features <--> distinct features linearly independent
# # show lift operation的injectivity
# # 这边只做了forward的情况，injective map 1-1
# # 纯linear，不保证injectivity, message passing, capture distinct featrures的速度慢
# # with lift operation, 信息传递更高效 -》 更能capture distinct features -〉 distinct 信息 - rank更快能高起来
# for dim in dim_list:
#     # x = torch.ones((data.num_nodes, dim), dtype=torch.float32)
#     h = torch.ones((data.num_nodes, dim), dtype=torch.float32)

#     distinct_rows_matrix = torch.unique(h, dim=0).float()

#     rank_of_distinct_matrix = torch.linalg.matrix_rank(distinct_rows_matrix, tol=tolerance)
#     print(f"\nRank of the distinct matrix: {rank_of_distinct_matrix.item()}")
#     # print(f"\nRank of the x matrix: {torch.linalg.matrix_rank(x, tol=1e-5).item()}")

#     # (index, rank) pairs
#     # rank_linear = [(0, rank_of_distinct_matrix.item())]
#     rank_non_linear = [(0, rank_of_distinct_matrix.item())]
#     distinct_node_feature = [distinct_rows_matrix.size(0)]
#     distinct_node_feature_x = [0]

#     for i in range(1, num_mp_layer+1):
#         # rank_linear.append((i, rank_linear[-1][1]))
#         rank_non_linear.append((i, rank_non_linear[-1][1]))

#         distinct_node_feature.append(distinct_node_feature[-1])
#         # distinct_node_feature.append(x_matrix.size(0))
#         distinct_node_feature_x.append(i)
#         # gnn = InjectiveMP(epsilon=math.pi - 3, hidden_dim=dim, linear=True)
#         # x = gnn(x, data.edge_index, data.edge_attr)

#         dln = InjectiveMP(eps=5**0.5/2, in_dim=dim, out_dim=dim, freeze=True) # hidden_dim=dim
#         h = dln(h, data.edge_index)
#         # print(torch.linalg.svdvals(h))
#         # print(torch.linalg.eigvals(h@h.T))
#         # x_matrix = torch.unique(x, dim=0).float()
#         h_matrix = torch.unique(h, dim=0).float()
        
#         # x_matrix = x.float()
#         # h_matrix = h.float()
#         print(f"H matrix: {h_matrix.size()}")
#         # rank_of_distinct_matrix_x = torch.linalg.matrix_rank(x_matrix, tol=1e-5)
#         rank_of_distinct_matrix_h = torch.linalg.matrix_rank(h_matrix, tol=tolerance)
#         print(f"Rank of the distinct matrix: {rank_of_distinct_matrix_h.item()}")
#         # rank_linear.append((i, min(rank_of_distinct_matrix_x.item(), h_matrix.size(0))))
#         rank_non_linear.append((i, min(rank_of_distinct_matrix_h.item(), h_matrix.size(0))))
#         distinct_node_feature.append(h_matrix.size(0))
#         distinct_node_feature_x.append(i)
#     # linear_res.append(rank_linear)
#     non_linear_res.append(rank_non_linear)

#     # print(f"Linear rank: {rank_linear}")
#     # print(f"Non-linear rank: {rank_non_linear}")
#     # print(f"Distinct node feature: {distinct_node_feature}") # might not be linearly independent

# res = non_linear_res

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set_theme(style="whitegrid") # Apply Seaborn's whitegrid style
# # Get a nice color palette from Seaborn
# palette = sns.color_palette("tab10", n_colors=len(res)+1)

# sns.set_theme(style="whitegrid") # Apply Seaborn styling
# plt.figure(figsize=(10, 6)) # Set the figure size
# plt.plot(distinct_node_feature_x, distinct_node_feature, linestyle='-', linewidth=1.5, color=palette[0], alpha=0.8, label="Num_distinct_node_feature")
# for i, non_linear_res_data in enumerate(res):
#     series_label = labels[i]
#     x = [point[0] for point in non_linear_res_data]
#     y = [point[1] for point in non_linear_res_data]
#     plt.plot(x, y, linestyle='--', linewidth=1.5, color=palette[i+1], alpha=0.8, label=series_label)

# # sns.lineplot(data=df, x='num_mp_layers', y='rank of distinct node feature matrix', hue='label', marker='o', linewidth=2.5, markersize=6, alpha=0.5, palette="viridis")

# # --- 4. Customize ---
# plt.title('')
# plt.xlabel('Number of Message Passing Layers')
# plt.ylabel('Distinct Features and Their Ranks')
# plt.xlim(0, num_mp_layer) # Set x-axis limits

# plt.legend() # Add legend (Seaborn usually adds one automatically)
# plt.tight_layout()
# # Save the figure to pdf
# plt.savefig(f'injective_tolerance_{tolerance}.pdf', format='pdf', bbox_inches='tight')
# plt.show()

def root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return parent_dir

def unique_rows_with_tolerance(x, per_item_tolerance=1e-4, total_tolerance=1e-4):
    rounded = torch.round(x / per_item_tolerance) * per_item_tolerance
    unique_rows = torch.unique(rounded, dim=0)
    return unique_rows.size(0)


def embedding_rank(x: torch.Tensor, tol=1e-15):
    x_unique = torch.unique(x, dim=0)
    x_unique = x_unique.double()
    print(f"H matrix: {x_unique.size()}")
    return x_unique.size(0), torch.linalg.matrix_rank(x_unique, tol=tol)
    # return unique_rows_with_tolerance(x_unique), torch.linalg.matrix_rank(x_unique, tol=tol)

def generate_expressive_power_plot(dataset_name='Cora', mp_depth=6, skip_conneciton=False, tolerance=1e-5, dim_list=[50]):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    # root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
    data_dir=f'{root_dir()}/data'
    data = load_dataset(data_dir=data_dir, dataset_name=dataset_name, filter=None if dataset_name != 'mnist' else 0)
    if dataset_name == 'mnist':
        data = Batch.from_data_list(data)

    # WL Test:
    _, wl_labels, distinct_features_each_iteration = wl_relabel(data, mp_depth)
    wl_groups = find_group(wl_labels)
    # Process the list for plotting purpose
    temp_list = []
    for i, k in enumerate(distinct_features_each_iteration):
        if i < len(distinct_features_each_iteration) - 1:
            temp_list.append(k)
        temp_list.append(k)
    distinct_features_each_iteration = temp_list
    
    dim_list = dim_list
    non_linear_res = []
    labels = [f"Non_linear_mp/dim={dim}" for dim in dim_list]
    distinct_node_feature_res = []
    distinct_node_feature_x = []

    edge_index = data.edge_index
    print(f'is undirected: {is_undirected(edge_index)}')
    edge_index = to_undirected(edge_index)

    data.to(device)
    edge_index = edge_index.to(device)

    for dim in dim_list:
        h = torch.ones((data.num_nodes, dim), dtype=torch.float32).to(device)

        distinct_rows_matrix = torch.unique(h, dim=0)

        rank_of_distinct_matrix = torch.linalg.matrix_rank(distinct_rows_matrix, tol=tolerance)
        print(f"\nRank of the distinct matrix: {rank_of_distinct_matrix.item()}")

        # (index, rank) pairs
        rank_non_linear = [(0, rank_of_distinct_matrix.item())]
        distinct_node_feature = [distinct_rows_matrix.size(0)]
        distinct_node_feature_x = [0]

        for i in range(1, mp_depth+1):
            rank_non_linear.append((i, rank_non_linear[-1][1]))

            distinct_node_feature.append(distinct_node_feature[-1])
            distinct_node_feature_x.append(i)

            dln = iMP(in_dim=dim, out_dim=dim, freeze=True, skip_connection=skip_conneciton, simple=True).to(device) # hidden_dim=dim
            h = dln(h, edge_index)
            # mp_groups = find_group(h)
            # h_matrix = torch.unique(h, dim=0).double()

            num_distinct_features, rank_of_distinct_matrix_h = embedding_rank(h, tol=tolerance)
            print(f"num distinct structures: {num_distinct_features}")
            print(f"Rank of the distinct matrix: {rank_of_distinct_matrix_h.item()}")
            rank_non_linear.append((i, min(rank_of_distinct_matrix_h.item(), num_distinct_features)))
            distinct_node_feature.append(num_distinct_features)
            distinct_node_feature_x.append(i)
        non_linear_res.append(rank_non_linear)
        distinct_node_feature_res.append(distinct_node_feature)

    res = non_linear_res

    sns.set_theme(style="whitegrid") # Apply Seaborn's whitegrid style
    # Get a nice color palette from Seaborn
    palette = sns.color_palette("tab10", n_colors=2*len(res)+1)

    sns.set_theme(style="whitegrid") # Apply Seaborn styling
    plt.figure(figsize=(10, 6)) # Set the figure size
    plt.plot(distinct_node_feature_x, distinct_features_each_iteration, linestyle='-.', linewidth=1.5, color=palette[0], alpha=1, label="WL Test")
    for i, distinct_features in enumerate(distinct_node_feature_res):
        plt.plot(distinct_node_feature_x, distinct_features, linestyle='-', linewidth=1.5, color=palette[i+1], alpha=0.8,
                 label=f"Num_distinct_node_feature/dim={dim_list[i]}")

    for i, non_linear_res_data in enumerate(res):
        series_label = labels[i]
        x = [point[0] for point in non_linear_res_data]
        y = [point[1] for point in non_linear_res_data]
        plt.plot(x, y, linestyle='--', linewidth=1.5, color=palette[i+len(res)+1], alpha=0.8, label=series_label)

    # --- 4. Customize ---
    plt.title(f'{dataset_name}, tolerance: {tolerance}')
    plt.xlabel('Number of Message Passing Layers')
    plt.ylabel('Distinct Features and Their Ranks')
    plt.xlim(0, mp_depth) # Set x-axis limits

    # plt.legend() # Add legend (Seaborn usually adds one automatically)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # Save the figure to pdf
    plt.savefig(f'{root_dir()}/injective_plot/injective_{dataset_name}_mp_{mp_depth}_tolerance_{tolerance}_{get_timestamp()}.png', bbox_inches='tight')
    # plt.show()


# def generate_expressive_power_plot_multi_graph(dataset_name='mnist', mp_depth=6, tolerance=1e-5, dim_list=[50]):
#     root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
#     data_dir=f'{root_dir}/data'
#     data = load_dataset(data_dir=data_dir, dataset_name=dataset_name)
#     data = Batch.from_data_list(data)

#     # WL Test:
#     # _, wl_labels, distinct_features_each_iteration = wl_relabel(data, mp_depth)
#     # wl_groups = find_group(wl_labels)
#     # # Process the list for plotting purpose
#     # temp_list = []
#     # for i, k in enumerate(distinct_features_each_iteration):
#     #     if i < len(distinct_features_each_iteration) - 1:
#     #         temp_list.append(k)
#     #     temp_list.append(k)
#     # distinct_features_each_iteration = temp_list
    

#     # dim_list = [50, 150, 300, 500, 1000, 2000, 4000, 8000]
#     dim_list = dim_list
#     non_linear_res = []
#     labels = [f"Non_linear_mp/dim={dim}" for dim in dim_list]
#     distinct_node_feature_res = []
#     distinct_node_feature_x = []

#     for dim in dim_list:
#         h = torch.ones((data.num_nodes, dim), dtype=torch.float32)

#         distinct_rows_matrix = torch.unique(h, dim=0).float()

#         rank_of_distinct_matrix = torch.linalg.matrix_rank(distinct_rows_matrix, tol=tolerance)
#         print(f"\nRank of the distinct matrix: {rank_of_distinct_matrix.item()}")

#         # (index, rank) pairs
#         rank_non_linear = [(0, rank_of_distinct_matrix.item())]
#         distinct_node_feature = [distinct_rows_matrix.size(0)]
#         distinct_node_feature_x = [0]

#         for i in range(1, mp_depth+1):
#             rank_non_linear.append((i, rank_non_linear[-1][1]))

#             distinct_node_feature.append(distinct_node_feature[-1])
#             distinct_node_feature_x.append(i)

#             dln = InjectiveMP(eps=5**0.5/2, in_dim=dim, out_dim=dim, freeze=True) # hidden_dim=dim
#             h = dln(h, data.edge_index)
#             mp_groups = find_group(h)
#             h_matrix = torch.unique(h, dim=0).float()
            
#             print(f"H matrix: {h_matrix.size()}")
#             rank_of_distinct_matrix_h = torch.linalg.matrix_rank(h_matrix, tol=tolerance)
#             print(f"Rank of the distinct matrix: {rank_of_distinct_matrix_h.item()}")
#             rank_non_linear.append((i, min(rank_of_distinct_matrix_h.item(), h_matrix.size(0))))
#             distinct_node_feature.append(h_matrix.size(0))
#             distinct_node_feature_x.append(i)
#         non_linear_res.append(rank_non_linear)
#         distinct_node_feature_res.append(distinct_node_feature)

#     res = non_linear_res

#     sns.set_theme(style="whitegrid") # Apply Seaborn's whitegrid style
#     # Get a nice color palette from Seaborn
#     palette = sns.color_palette("tab10", n_colors=2*len(res)+1)

#     sns.set_theme(style="whitegrid") # Apply Seaborn styling
#     plt.figure(figsize=(10, 6)) # Set the figure size
#     plt.plot(distinct_node_feature_x, distinct_features_each_iteration, linestyle='-.', linewidth=1.5, color=palette[0], alpha=1, label="WL Test")
#     for i, distinct_features in enumerate(distinct_node_feature_res):
#         plt.plot(distinct_node_feature_x, distinct_features, linestyle='-', linewidth=1.5, color=palette[i+1], alpha=0.8,
#                  label=f"Num_distinct_node_feature/dim={dim_list[i]}")

#     for i, non_linear_res_data in enumerate(res):
#         series_label = labels[i]
#         x = [point[0] for point in non_linear_res_data]
#         y = [point[1] for point in non_linear_res_data]
#         plt.plot(x, y, linestyle='--', linewidth=1.5, color=palette[i+len(res)+1], alpha=0.8, label=series_label)

#     # --- 4. Customize ---
#     plt.title(f'{dataset_name}, tolerance: {tolerance}')
#     plt.xlabel('Number of Message Passing Layers')
#     plt.ylabel('Distinct Features and Their Ranks')
#     plt.xlim(0, mp_depth) # Set x-axis limits

#     # plt.legend() # Add legend (Seaborn usually adds one automatically)
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.tight_layout()
#     # Save the figure to pdf
#     plt.savefig(f'{root_dir}/injective_plot/injective_{dataset_name}_mp_{mp_depth}_tolerance_{tolerance}.pdf', format='pdf', bbox_inches='tight')
#     plt.show()

from main import run
def generate_expressive_power_plot_with_training(dataset_name='Cora', mp_depth=6, tolerance=1e-5, skip_connection=False, dropout=0,
                                                 dim_list=[50], num_fl_layers=2, fl_hidden_dim=128,
                                                 epsilon=5**0.5/2, optimizer_lr=0.01, total_epoch=200):
    '''
    First train our model on a dataset, and then do the forward pass to evaluate the injective layers'
    expressive power and ranks
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # for each dimension mentioned in the dim_list, we need to train a model.
    # root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
    data_dir=f'{root_dir()}/data'
    data = load_dataset(data_dir=data_dir, dataset_name=dataset_name, filter=None if dataset_name != 'mnist' else 0).to(device)
    if dataset_name == 'mnist':
        data = Batch.from_data_list(data)

    # WL Test:
    _, wl_labels, distinct_features_each_iteration = wl_relabel(data, mp_depth)
    # wl_groups = find_group(wl_labels)
    # Process the list for plotting purpose
    temp_list = []
    for i, k in enumerate(distinct_features_each_iteration):
        if i < len(distinct_features_each_iteration) - 1:
            temp_list.append(k)
        temp_list.append(k)
    distinct_features_each_iteration = temp_list

    dim_list = dim_list
    non_linear_res = []
    labels = [f"Non_linear_mp/dim={dim}" for dim in dim_list]
    distinct_node_feature_res = []
    distinct_node_feature_x = []

    edge_index = data.edge_index
    # print(f'is undirected: {is_undirected(edge_index)}')
    edge_index = to_undirected(edge_index)

    # Do the training
    loss_func = 'CrossEntropyLoss'
    for dim in dim_list:
        best_val, best_test, model, _, _, _ = run(dataset_name, mp_depth, num_fl_layers, dim,
                                         fl_hidden_dim, epsilon, optimizer_lr, loss_func,
                                         total_epoch, index=0, freeze=False, save_model=False, skip_connection=skip_connection, dropout=dropout, folder_name_suffix="check_dropout_after_mp")

        mp_model = MPOnlyModel(model)
        h = torch.ones((data.num_nodes, data.x.shape[1]), dtype=torch.float32).to(device)
        distinct_rows_matrix = torch.unique(h, dim=0).float()

        rank_of_distinct_matrix = torch.linalg.matrix_rank(distinct_rows_matrix, tol=tolerance)
        print(f"\nRank of the distinct matrix: {rank_of_distinct_matrix.item()}")

        # (index, rank) pairs
        rank_non_linear = [(0, rank_of_distinct_matrix.item())]
        distinct_node_feature = [distinct_rows_matrix.size(0)]
        distinct_node_feature_x = [0]

        for i in range(1, mp_depth+1):
            rank_non_linear.append((i, rank_non_linear[-1][1]))

            distinct_node_feature.append(distinct_node_feature[-1])
            distinct_node_feature_x.append(i)

            h = mp_model(h, edge_index, i-1)
            # mp_groups = find_group(h)

            num_distinct_features, rank_of_distinct_matrix_h = embedding_rank(h, tol=tolerance)
            print(f"num distinct structures: {num_distinct_features}")
            print(f"Rank of the distinct matrix: {rank_of_distinct_matrix_h.item()}")
            rank_non_linear.append((i, min(rank_of_distinct_matrix_h.item(), num_distinct_features)))            
            distinct_node_feature.append(num_distinct_features)
            distinct_node_feature_x.append(i)
        non_linear_res.append(rank_non_linear)
        distinct_node_feature_res.append(distinct_node_feature)

        # ---- Clean up after each model ----
        del mp_model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    res = non_linear_res

    sns.set_theme(style="whitegrid") # Apply Seaborn's whitegrid style
    # Get a nice color palette from Seaborn
    palette = sns.color_palette("tab10", n_colors=2*len(res)+1)

    sns.set_theme(style="whitegrid") # Apply Seaborn styling
    plt.figure(figsize=(10, 6)) # Set the figure size
    plt.plot(distinct_node_feature_x, distinct_features_each_iteration, linestyle='-.', linewidth=1.5, color=palette[0], alpha=1, label="WL Test")
    for i, distinct_features in enumerate(distinct_node_feature_res):
        plt.plot(distinct_node_feature_x, distinct_features, linestyle='-', linewidth=1.5, color=palette[i+1], alpha=0.8,
                 label=f"Num_distinct_node_feature/dim={dim_list[i]}")

    for i, non_linear_res_data in enumerate(res):
        series_label = labels[i]
        x = [point[0] for point in non_linear_res_data]
        y = [point[1] for point in non_linear_res_data]
        plt.plot(x, y, linestyle='--', linewidth=1.5, color=palette[i+len(res)+1], alpha=0.8, label=series_label)

    # --- 4. Customize ---
    plt.title(f'{dataset_name}, tolerance: {tolerance}')
    plt.xlabel('Number of Message Passing Layers')
    plt.ylabel('Distinct Features and Their Ranks')
    plt.xlim(0, mp_depth) # Set x-axis limits

    # plt.legend() # Add legend (Seaborn usually adds one automatically)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # Save the figure to pdf
    plt.savefig(f'{root_dir()}/injective_plot_my_dropout/train_injective_{dataset_name}_mp_{mp_depth}_tolerance_{tolerance}_dropout_{dropout}_{get_timestamp()}.png', bbox_inches='tight')
    # plt.show()




def generate_expressive_power_plot_transfer_learning(dataset_name='Cora', mp_depth=6, tolerance=1e-5, dim_list=[50]):
    root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
    data_dir=f'{root_dir}/data'
    data = load_dataset(data_dir=data_dir, dataset_name=dataset_name, filter=None if dataset_name != 'mnist' else 0)
    if dataset_name == 'mnist':
        data = Batch.from_data_list(data)

    # WL Test:
    _, wl_labels, distinct_features_each_iteration = wl_relabel(data, mp_depth)
    wl_groups = find_group(wl_labels)
    # Process the list for plotting purpose
    temp_list = []
    for i, k in enumerate(distinct_features_each_iteration):
        if i < len(distinct_features_each_iteration) - 1:
            temp_list.append(k)
        temp_list.append(k)
    distinct_features_each_iteration = temp_list
    

    # dim_list = [50, 150, 300, 500, 1000, 2000, 4000, 8000]
    dim_list = dim_list
    non_linear_res = []
    labels = [f"Non_linear_mp/dim={dim}" for dim in dim_list]
    distinct_node_feature_res = []
    distinct_node_feature_x = []


    num_fl_layers = 2 # number of mlp layer
    mp_hidden_dim = 50
    fl_hidden_dim = 128
    epsilon = 5**0.5/2
    d = data.x.shape[1]
    c = data.y.max().item() + 1

    model = DecoupleModel (
        in_dim=d,
        out_dim=c,
        mp_width=mp_hidden_dim,
        fl_width=fl_hidden_dim,
        num_mp_layers = mp_depth,
        num_fl_layers = num_fl_layers,
        eps=epsilon,
        freeze=True
    )
    model.load_state_dict(torch.load('/Users/yifanguan/gnn_research/GraphLearning/saved_models/model_weights.pth'))
    mp_model = MPOnlyModel(model)

    for dim in dim_list:
        h = torch.ones((data.num_nodes, d), dtype=torch.float32)

        distinct_rows_matrix = torch.unique(h, dim=0).float()

        rank_of_distinct_matrix = torch.linalg.matrix_rank(distinct_rows_matrix, tol=tolerance)
        print(f"\nRank of the distinct matrix: {rank_of_distinct_matrix.item()}")

        # (index, rank) pairs
        rank_non_linear = [(0, rank_of_distinct_matrix.item())]
        distinct_node_feature = [distinct_rows_matrix.size(0)]
        distinct_node_feature_x = [0]

        for i in range(1, mp_depth+1):
            rank_non_linear.append((i, rank_non_linear[-1][1]))

            distinct_node_feature.append(distinct_node_feature[-1])
            distinct_node_feature_x.append(i)

            h = mp_model(h, data.edge_index, i-1)
            mp_groups = find_group(h)
            h_matrix = torch.unique(h, dim=0).float()
            
            print(f"H matrix: {h_matrix.size()}")
            rank_of_distinct_matrix_h = torch.linalg.matrix_rank(h_matrix, tol=tolerance)
            print(f"Rank of the distinct matrix: {rank_of_distinct_matrix_h.item()}")
            rank_non_linear.append((i, min(rank_of_distinct_matrix_h.item(), h_matrix.size(0))))
            distinct_node_feature.append(h_matrix.size(0))
            distinct_node_feature_x.append(i)
        non_linear_res.append(rank_non_linear)
        distinct_node_feature_res.append(distinct_node_feature)

    res = non_linear_res

    sns.set_theme(style="whitegrid") # Apply Seaborn's whitegrid style
    # Get a nice color palette from Seaborn
    palette = sns.color_palette("tab10", n_colors=2*len(res)+1)

    sns.set_theme(style="whitegrid") # Apply Seaborn styling
    plt.figure(figsize=(10, 6)) # Set the figure size
    plt.plot(distinct_node_feature_x, distinct_features_each_iteration, linestyle='-.', linewidth=1.5, color=palette[0], alpha=1, label="WL Test")
    for i, distinct_features in enumerate(distinct_node_feature_res):
        plt.plot(distinct_node_feature_x, distinct_features, linestyle='-', linewidth=1.5, color=palette[i+1], alpha=0.8,
                 label=f"Num_distinct_node_feature/dim={dim_list[i]}")

    for i, non_linear_res_data in enumerate(res):
        series_label = labels[i]
        x = [point[0] for point in non_linear_res_data]
        y = [point[1] for point in non_linear_res_data]
        plt.plot(x, y, linestyle='--', linewidth=1.5, color=palette[i+len(res)+1], alpha=0.8, label=series_label)

    # --- 4. Customize ---
    plt.title(f'{dataset_name}, tolerance: {tolerance}')
    plt.xlabel('Number of Message Passing Layers')
    plt.ylabel('Distinct Features and Their Ranks')
    plt.xlim(0, mp_depth) # Set x-axis limits

    # plt.legend() # Add legend (Seaborn usually adds one automatically)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # Save the figure to pdf
    plt.savefig(f'{root_dir}/injective_plot/injective_{dataset_name}_mp_{mp_depth}_tolerance_{tolerance}_{get_timestamp()}.png', bbox_inches='tight')
    plt.show()

