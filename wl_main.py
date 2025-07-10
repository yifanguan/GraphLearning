from utils.wl_test import wl_relabel, wl_relabel_multigraph
from utils.dataset import load_dataset 
from utils.wl_test import networkx_wl_relabel, networkx_wl_relabel_multi_graphs

# texas has 129 distinct features
# cornell5 has 18577 distinct features
# mnist 0 digit has 517313 distinct features

root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
data_dir=f'{root_dir}/data'
data = load_dataset(data_dir=data_dir, dataset_name='cora')
# print(f'texas data: {data}')

# k, _, distinct_features_each_iteration = wl_relabel(data, 30)


# root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
# data_dir=f'{root_dir}/data'
# data = load_dataset(data_dir=data_dir, dataset_name='texas')
# print(f'cornell5 data: {data}')

# k, _, distinct_features_each_iteration = wl_relabel(data, 20)

# k, node_labels = networkx_wl_relabel(data, 20)
# print(node_labels[0])


root_dir = '/Users/yifanguan/gnn_research/GraphLearning'
data_dir=f'{root_dir}/data'
data = load_dataset(data_dir=data_dir, dataset_name='mnist', filter=0)
# print(f'minist data: {data}')

# k, _, distinct_features_each_iteration = wl_relabel(data, 20)

# k, _ = networkx_wl_relabel_multi_graphs(data, 20)
k, _, _ = wl_relabel_multigraph(data, 20)
