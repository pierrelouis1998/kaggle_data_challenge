import cvxpy as cp
import numpy as np
from typing import List

from kernel_utils import centered_gram_matrix


def kernel_pca(K: np.ndarray, n_components=2) -> np.ndarray:
    """Perform kernel PCA"""
    n = K.shape[0]
    centered_gram = centered_gram_matrix(K)
    vals, vect = np.linalg.eigh(centered_gram)
    vals = vals[::-1]
    vect = vect[:, ::-1]
    if abs(vals[n_components]) < 1e-6:
        raise ValueError(f"0 eigen value")
    # print(vals)
    return np.vstack([vect[:, i] / np.sqrt(vals[i]) for i in range(n_components)]).T
