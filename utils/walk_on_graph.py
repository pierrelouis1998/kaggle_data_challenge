from typing import List, Callable
from tqdm import tqdm
import networkx as nx
import numpy as np
import sys

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
    A = nx.adjacency_matrix(G)
    n_node = nx.number_of_nodes(G)
    K = np.ones(n_node).T @ A ** n @ np.ones(n_node)
    return np.asarray(K).reshape(1)


def compute_kernel_on_dataset(dataset: List[nx.Graph], kernel: Callable, kernel_kwargs = None, disable_tqdm: bool = False) -> np.ndarray:
    """
    Compute Kernel evaluation on a dataset
    :param disable_tqdm:
    :param kernel_kwargs:
    :param dataset:
    :param kernel:
    :return:
    """
    K = np.zeros((len(dataset),len(dataset)))
    for i, G1 in tqdm(enumerate(dataset), desc=f"Evaluating kernel on {len(dataset)} graph", unit="Graph", disable=disable_tqdm, total=len(dataset)):
        for j, G2 in enumerate(dataset[i:]):
            res = kernel(G1, G2, **kernel_kwargs)
            K[i,j] = res
            K[j,i] = res
    return K