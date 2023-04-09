import networkx as nx
import numpy as np

from graph_utils import graph_product_node_labels


def nth_order_walk_kernel(G1: nx.Graph, G2: nx.Graph, n: int = 5):
    """
    Compute nth order walk kernel
    K(G1,G2) = 1.T x A^n x 1
    :param G1: First graph
    :param G2: Second graph
    :param n: length of the walk
    :return: K(G1,G2)
    """
    G = graph_product_node_labels(G1, G2)
    # G = nx.tensor_product(G1,G2)
    A = nx.adjacency_matrix(G)
    K = np.sum(A**n)
    return np.asarray(K).astype(int).reshape(1)
