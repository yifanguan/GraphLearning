import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import sys, os
# Add parent folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dln import InjectiveMP
from utils.wl_test import wl_relabel, find_group
from utils.dataset import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns


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



def generate_expressive_power_plot(dataset_name='Cora', mp_depth=6, tolerance=1e-5, dim_list=[50]):
    root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
    data_dir=f'{root_dir}/data'
    data = load_dataset(data_dir=data_dir, dataset_name=dataset_name)

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

    for dim in dim_list:
        h = torch.ones((data.num_nodes, dim), dtype=torch.float32)

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

            dln = InjectiveMP(eps=5**0.5/2, in_dim=dim, out_dim=dim, freeze=True) # hidden_dim=dim
            h = dln(h, data.edge_index)
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
    plt.savefig(f'{root_dir}/injective_plot/injective_{dataset_name}_mp_{mp_depth}_tolerance_{tolerance}.pdf', format='pdf', bbox_inches='tight')
    plt.show()
