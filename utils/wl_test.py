# WL Test Util used to find distinct neighborhood
import networkx as nx
from hashlib import md5
# from collections import Counter
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import torch
from networkx.algorithms.graph_hashing import _hash_label, _neighborhood_aggregate
from collections import Counter, defaultdict
from torch_geometric.data import Batch


def wl_hash(node_label, neighbor_labels):
    # Sort neighbor labels and concatenate with node label
    combined = str(node_label) + "|".join(sorted(map(str, neighbor_labels)))
    hash_str = md5(combined.encode()).hexdigest()

    return hash_str

# Reassign labels for nodes, only change old label to new label for new distinct nodes
def reassign_label(old_labels, new_labels):
    old_color_assignments = {} # label -> list of nodes
    for node, label in old_labels.items():
        if label not in old_color_assignments:
            old_color_assignments[label] = []
        old_color_assignments[label].append(node)

    new_color_assignments = {} # label -> list of nodes
    for node, label in new_labels.items():
        if label not in new_color_assignments:
            new_color_assignments[label] = []
        new_color_assignments[label].append(node)
    
    res = old_labels.copy()
    used_old_label = set()
    for new_label, new_nodes in new_color_assignments.items():
        count = 0
        for old_label, old_nodes in old_color_assignments.items():
            if set(new_nodes).issubset(old_nodes):
                count += 1
                if old_label not in used_old_label:
                    used_old_label.add(old_label)
                    for node in new_nodes:
                        res[node] = old_label
                else:
                    # need a new label instead of reusing the old one
                    for node in new_nodes:
                        res[node] = new_label
        assert count == 1

    return res

def count_distinct(labels):
    return len(set(labels.values()))
    # return len(Counter(labels.values()).values())

# WL Test Initialization
# WL test starts by assigning each node a label.
# If a label is not provided, we often use something like the node degree as a proxy.
# These labels are then updated over multiple iterations to reflect the structure of the graph.
def wl_relabel(graph, h: int):
    """
    Perform h iterations of the Weisfeiler-Lehman Test relabeling on the input graph.
    Initially, each graph node use its degree as its initial state
    This function is used to find the number of distinct neighborhood structures: k in a graph.

    Args:
        graph: A NetworkX graph with or without node labels.
        h: Number of WL iterations.

    Returns:
        k: number of distinct neighborhood structures
        labels: dict for all nodes' labels
        distinct_features_each_iteration: a list containing total number of distinct structures after each iteration of WL test
    """
    graph = to_networkx(graph, to_undirected=True)
    labels = {} # node -> label
    for node in graph.nodes():
        # labels[node] = graph.degree[node]
        labels[node] = 0
    
    is_converge = False
    k = count_distinct(labels)
    distinct_features_each_iteration = [k]
    # print('k is {}', k)
    for i in range(h):
        new_labels = {}
        for node in graph.nodes():
            new_label = wl_hash(labels[node], [labels[neighbor] for neighbor in graph.neighbors(node)])
            new_labels[node] = new_label
        # new_labels = reassign_label(labels, new_labels)
        new_k = count_distinct(new_labels)

        if new_k == k:
            # print("Converge on {} distinct structures".format(k))
            is_converge = True

        labels = new_labels
        k = new_k
        distinct_features_each_iteration.append(k)
        print("After Iteration {}, {} distinct structures, is converge? {}".format(i, k, is_converge))

    return k, labels, distinct_features_each_iteration

def wl_train_test_ood(labels, train_idx, test_idx):
    '''
    Helper function used to evaluate ood.
    Return number of distinct features in train, number of distinct features in test,
    and number of distinct features exists in both train and test.
    '''
    train_idx = [idx.item() for idx in train_idx]
    test_idx = [idx.item() for idx in test_idx]
    train_labels = {}
    test_labels = {}
    for idx in train_idx:
        assert idx in labels, "missing train node label"
        train_labels[idx] = labels[idx]
    for idx in test_idx:
        assert idx in labels, "missing test node label"
        test_labels[idx] = labels[idx]
    train_distinct_set = set(train_labels.values())
    test_distinct_set = set(test_labels.values())
    train_test_intersection = train_distinct_set & test_distinct_set

    return len(train_distinct_set), len(test_distinct_set), len(train_test_intersection)


def find_group(labels):
    '''
    Find group of nodes with the same color
    '''
    groups = {}
    if isinstance(labels, dict):
        for node, label in labels.items():
            if label not in groups:
                groups[label] = []
            groups[label].append(node)
        groups = list(groups.values())
    if isinstance(labels, torch.Tensor):
        groups = []
        h_matrix = labels
        for i, h in enumerate(h_matrix):
            found = False
            for group in groups:
                node = group[0]
                if torch.equal(h_matrix[node], h):
                    found = True
                    group.append(i)
            if not found:
                groups.append([i])
    return groups

def wl_relabel_multigraph(data, h: int):
    if not isinstance(data, Batch):
        batch = Batch.from_data_list(data)
    k, labels, distinct_features_each_iteration = wl_relabel(batch, h)
    return k, labels, distinct_features_each_iteration


def weisfeiler_lehman_subgraph_hashes(
    G,
    edge_attr=None,
    node_attr=None,
    iterations=3,
    digest_size=16,
    include_initial_labels=False,
):
    """
    Copy from networkx weisfeiler_lehman_subgraph_hashes
    Modify it for node feature initialization 
    """

    def weisfeiler_lehman_step(G, labels, node_subgraph_hashes, edge_attr=None):
        """
        Apply neighborhood aggregation to each node
        in the graph.
        Computes a dictionary with labels for each node.
        Appends the new hashed label to the dictionary of subgraph hashes
        originating from and indexed by each node in G
        """
        new_labels = {}
        for node in G.nodes():
            label = _neighborhood_aggregate(G, node, labels, edge_attr=edge_attr)
            hashed_label = _hash_label(label, digest_size)
            new_labels[node] = hashed_label
            node_subgraph_hashes[node].append(hashed_label)
        return new_labels
    
    def _init_node_labels(G, edge_attr, node_attr):
        if node_attr:
            return {u: str(dd[node_attr]) for u, dd in G.nodes(data=True)}
        elif edge_attr:
            return {u: "" for u in G}
        else:
            return {u: "1" for u in G}

    node_labels = _init_node_labels(G, edge_attr, node_attr)
    if include_initial_labels:
        node_subgraph_hashes = {
            k: [_hash_label(v, digest_size)] for k, v in node_labels.items()
        }
    else:
        node_subgraph_hashes = defaultdict(list)

    for _ in range(iterations):
        node_labels = weisfeiler_lehman_step(
            G, node_labels, node_subgraph_hashes, edge_attr
        )

    return dict(node_subgraph_hashes)



def networkx_wl_relabel(graph: Data, h: int):
    '''
    single graph version
    '''
    graph = to_networkx(graph, to_undirected=True)
    node_labels = weisfeiler_lehman_subgraph_hashes(graph, iterations=h, include_initial_labels=False)
    k = 0
    for i in range(h):
        node_labels_at_iteration_i = set()
        for node, labels in node_labels.items():
            node_labels_at_iteration_i.add(labels[i])
        k = len(node_labels_at_iteration_i)
        print(f'iteration {i}: has {k} distinct structures')
    return k, node_labels


def networkx_wl_relabel_multi_graphs(dataset, h: int):
    '''
    Dataset containing multiple graphs version
    '''
    if not isinstance(dataset, Batch):
        batch = Batch.from_data_list(dataset)
    graph = to_networkx(batch, to_undirected=True)
    node_labels = weisfeiler_lehman_subgraph_hashes(graph, iterations=h, include_initial_labels=False)
    k = 0
    for i in range(h):
        node_labels_at_iteration_i = set()
        for node, labels in node_labels.items():
            node_labels_at_iteration_i.add(labels[i])
        k = len(node_labels_at_iteration_i)
        print(f'iteration {i}: has {k} distinct structures')
    return k, node_labels
