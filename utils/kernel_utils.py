from typing import List, Callable

import networkx as nx
import numpy as np
from tqdm import tqdm


def centered_gram_matrix(K: np.ndarray) -> np.ndarray:
    """Compute centered gram matrix from data"""
    n = K.shape[0]
    return K - np.mean(K, axis=1)[:, None] - np.mean(K, axis=0)[None, :] + (1 / n ** 2) * np.sum(K)


def compute_kernel_on_dataset(dataset: List[nx.Graph], kernel: Callable, kernel_kwargs=None,
                              disable_tqdm: bool = False) -> np.ndarray:
    """
    Compute Kernel evaluation on a dataset
    :param disable_tqdm:
    :param kernel_kwargs:
    :param dataset:
    :param kernel:
    :return:
    """
    K = np.zeros((len(dataset), len(dataset)))
    for i, G1 in tqdm(enumerate(dataset), desc=f"Evaluating kernel on {len(dataset)} graph", unit="Graph",
                      disable=disable_tqdm, total=len(dataset)):
        for j, G2 in enumerate(dataset[i:]):
            res = kernel(G1, G2, **kernel_kwargs)
            K[i, j] = res
            K[j, i] = res
    return K
