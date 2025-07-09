# WL Test Util used to find distinct neighborhood
import networkx as nx
from hashlib import md5
# from collections import Counter
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


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
def wl_relabel(graph: Data, h: int):
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
        new_labels = reassign_label(labels, new_labels)
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
