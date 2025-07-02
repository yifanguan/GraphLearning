# WL Test Util used to find distinct neighborhood
import networkx as nx
from hashlib import md5
from collections import Counter
import importlib.resources
import struct
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data


def wl_hash(node_label, neighbor_labels):
    # Sort neighbor labels and concatenate with node label
    combined = str(node_label) + "|".join(sorted(map(str, neighbor_labels)))
    hash_str = md5(combined.encode()).hexdigest()
    # hash_int = int(hash_str, 16)
    # # Normalize to [0, 1]
    # max_int = int("f" * 32, 16)  # max possible md5 value = 2^128 - 1
    # normalized = hash_int / max_int
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
    return len(Counter(labels.values()).values())

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
    """
    graph = to_networkx(graph, to_undirected=True)
    labels = {} # node -> label
    for node in graph.nodes():
        labels[node] = graph.degree[node]

    # gloabl_labels = labels.copy()
    # print(labels)
    
    is_converge = False
    k = count_distinct(labels)
    print('k is {}', k)
    for i in range(h):
        new_labels = {}
        for node in graph.nodes():
            new_label = wl_hash(labels[node], [labels[neighbor] for neighbor in graph.neighbors(node)])
            new_labels[node] = new_label
        # new_k = count_distinct(new_labels)
        new_labels = reassign_label(labels, new_labels)
        new_k = count_distinct(new_labels)
        # print("new_k: {}".format(new_k))
        # print("new_k_2: {}".format(new_k_2))
        # print("new_k: {}, new_k_2: {}".format(new_k, new_k_2))

        if new_k == k:
            print("Converge on {} distinct structures".format(k))
            is_converge = True

        labels = new_labels
        # gloabl_labels.update(labels)
        k = new_k
        print("After Iteration {}, {} distinct structures, is converge? {}".format(i, k, is_converge))

    return k

# def wl_test(G1, G2, h: int, nodes1=None, nodes2=None) -> bool:
#     """
#     Perform the Weisfeiler-Lehman Test for graph isomorphism.
#     Args:
#         G1: A NetworkX graph with or without node labels.
#         G2: A NetworkX graph with or without node labels.
#         nodes: initial nodes' states i.e. initial features

#     Returns:
#         True if the graphs are not isomorphic, False if they might be isomorphic.
#     """
#     # initialize nodes features

#     # perform testing
#     g1_color, _ = wl_relabel(G1, nodes1, use_degree=False, h=h)
#     g2_color, _ = wl_relabel(G2, nodes2, use_degree=False, h=h)
#     return g1_color == g2_color



# https://github.com/networkx/networkx/blob/main/networkx/algorithms/isomorphism/tests/test_isomorphism.py
def build_graph():
    G1 = nx.Graph()
    G2 = nx.Graph()
    G3 = nx.Graph()
    G4 = nx.Graph()
    G5 = nx.Graph()
    G6 = nx.Graph()
    G1.add_edges_from([[1, 2], [1, 3], [1, 5], [2, 3]])
    G2.add_edges_from([[10, 20], [20, 30], [10, 30], [10, 50]])
    G3.add_edges_from([[1, 2], [1, 3], [1, 5], [2, 5]])
    G4.add_edges_from([[1, 2], [1, 3], [1, 5], [2, 4]])
    G5.add_edges_from([[1, 2], [1, 3]])
    G6.add_edges_from([[10, 20], [20, 30], [10, 30], [10, 50], [20, 50]])

    return G1, G2

class TestWikipediaExample:
    # Source: https://en.wikipedia.org/wiki/Graph_isomorphism

    # Nodes 'a', 'b', 'c' and 'd' form a column.
    # Nodes 'g', 'h', 'i' and 'j' form a column.
    g1edges = [
        ["a", "g"],
        ["a", "h"],
        ["a", "i"],
        ["b", "g"],
        ["b", "h"],
        ["b", "j"],
        ["c", "g"],
        ["c", "i"],
        ["c", "j"],
        ["d", "h"],
        ["d", "i"],
        ["d", "j"],
    ]

    # Nodes 1,2,3,4 form the clockwise corners of a large square.
    # Nodes 5,6,7,8 form the clockwise corners of a small square
    g2edges = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 5],
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 8],
    ]

    def test_graph(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        g1.add_edges_from(self.g1edges)
        g2.add_edges_from(self.g2edges)

        return g1, g2



class TestVF2GraphDB:
    # https://web.archive.org/web/20090303210205/http://amalfi.dis.unina.it/graph/db/

    @staticmethod
    def create_graph(filename):
        """Creates a Graph instance from the filename."""

        # The file is assumed to be in the format from the VF2 graph database.
        # Each file is composed of 16-bit numbers (unsigned short int).
        # So we will want to read 2 bytes at a time.

        # We can read the number as follows:
        #   number = struct.unpack('<H', file.read(2))
        # This says, expect the data in little-endian encoding
        # as an unsigned short int and unpack 2 bytes from the file.

        fh = open(filename, mode="rb")

        # Grab the number of nodes.
        # Node numeration is 0-based, so the first node has index 0.
        nodes = struct.unpack("<H", fh.read(2))[0]

        graph = nx.Graph()
        for from_node in range(nodes):
            # Get the number of edges.
            edges = struct.unpack("<H", fh.read(2))[0]
            for edge in range(edges):
                # Get the terminal node.
                to_node = struct.unpack("<H", fh.read(2))[0]
                graph.add_edge(from_node, to_node)

        fh.close()
        return graph

    def test_graph(self):
        head = importlib.resources.files("networkx.algorithms.isomorphism.tests")
        g1 = self.create_graph(head / "iso_r01_s80.A99")
        g2 = self.create_graph(head / "iso_r01_s80.B99")
        return g1, g2

if __name__ == "__main__":
    # Build graph
    # Create an empty graph
    # G1 = nx.Graph()

    # # Add nodes and edges
    # G1.add_nodes_from([1, 2, 3])
    # G1.add_edges_from([(1, 2), (2, 3), (3, 1)])

    # View nodes and edges
    # print(G1.nodes())  # [1, 2, 3]
    # print(G1.edges())  # [(1, 2), (2, 3), (3, 1)]

    # Create two isomorphic graphs (triangle)
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2), (2, 0)])

    # G2 = nx.Graph()
    # G2.add_edges_from([(10, 11), (11, 12), (12, 10)])


    # G1, G2 = TestVF2GraphDB().test_graph()
    # Run WL test
    # (1) use node degree
    # labels1, h1 = wl_relabel(G1, None, True, h=3)
    # labels2, h2 = wl_relabel(G2, None, True, h=3)

    # (2) use intial node embedding
    # nodes1 = {}
    # for node in G1.nodes():
    #     nodes1[node] = 0
    # nodes2 = {}
    # labels1, h1 = wl_relabel(G1, nodes1, False, h=3)
    # labels2, h2 = wl_relabel(G2, None, True, h=3)

    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (0, 2), (1, 2), (1,3), (2,3)])
    wl_relabel(from_networkx(G1), 10)


    # print("G1 WL Histogram:", h1)
    # print("G2 WL Histogram:", h2)
    # print("Are histograms equal? ", h1 == h2)
    # print("Is G1 isomorphism to G2? {}".format(set(Counter(labels1.values()).values()) == set(Counter(labels2.values()).values())))
