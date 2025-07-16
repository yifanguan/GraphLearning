from utils.wl_test import wl_relabel, wl_relabel_multigraph, adam_wl_refinement
from utils.dataset import load_dataset 
from utils.wl_test import networkx_wl_relabel, networkx_wl_relabel_multi_graphs

# mnist 0 digit has 517313 distinct features
# texas has 129 distinct features, 4 wl iterations
# cornell5 has 18577 distinct features, 3 wl iterations
# amazon-photo has 7460 distinct features, 3 wl iterations
# amazon-computers has 13349 distinct features, 4 wl iterations
# coauthor-cs has 17891 distinct features, 5 wl iterations
# coauthor-physics has 33661 distinct features, 5 wl iterations
# wikics has 10862 distinct features, 4 wl iterations
# roman-empire 22661 distinct features, 8 wl iterations
# amazon-ratings 19071 distinct features, 5 wl iterations
# minesweeper 1275 distinct features, 50 wl iterations
# tolokers 11595 distinct features, 4 wl iterations
# questions 27899 distinct features, 6 wl iterations
# squirrel 2197 distinct features, 4 wl iterations
# chameleon 808 distinct features, 3 wl iterations
# ogbn-arxiv 162564 distinct features, 7 wl iterations
# ogbn-proteins 128261 distinct features, 3 wl iterations

# ogbn-products 162564 distinct features, 7 wl iterations



# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='texas')
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(f'texas data: {data}')

# k, _, distinct_features_each_iteration = wl_relabel(data, 30)


# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='cornell5')
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(f'cornell5 data: {data}')

# k, _, distinct_features_each_iteration = wl_relabel(data, 20)

# k, node_labels = networkx_wl_relabel(data, 20)
# print(node_labels[0])


# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='mnist', filter=0)
# # print(f'minist data: {data}')

# # k, _, distinct_features_each_iteration = wl_relabel(data, 20)

# # k, _ = networkx_wl_relabel_multi_graphs(data, 20)
# k, _, _ = wl_relabel_multigraph(data, 20)



# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='amazon-photo')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)


# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='amazon-computers')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)


# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='coauthor-cs')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)


# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='coauthor-physics')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)


# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='wikics')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)


# 'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'
# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='roman-empire')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)

# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='amazon-ratings')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)

# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='minesweeper')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 50)
# print(k)

# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='minesweeper')
# k, node_labels = networkx_wl_relabel(data, 50)
# print(k)
# print(node_labels[0])

# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='tolokers')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 20)
# print(k)

# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='questions')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 20)
# print(k)


# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='squirrel')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 20)
# print(k)


# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='chameleon')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 20)
# print(k)


# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='ogbn-arxiv')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)


# TODO: run, it is too large
# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='ogbn-products')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)

# TODO: run, it is too large
# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='pokec')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)

# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='ogbn-proteins')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)


# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='REDDIT-BINARY')
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)



# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='reddit')
# adam_wl_refinement(data, data.edge_index)
# print(data)
# k, _, distinct_features_each_iteration = wl_relabel(data, 30)
# print(k)

root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
data_dir=f'{root_dir}/data'
data = load_dataset(data_dir=data_dir, dataset_name='citeseer')
k, _, distinct_features_each_iteration = wl_relabel(data, 30)
