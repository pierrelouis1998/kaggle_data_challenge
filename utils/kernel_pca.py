import cvxpy as cp
import numpy as np
from typing import List

from kernel_utils import centered_gram_matrix


def kernel_pca(K: np.ndarray, n_component: int = 2) -> List[np.ndarray]:
    """Perform kernel PCA"""
    n = K.shape[0]
    centered_gram = centered_gram_matrix(K)
    components = []
    for idx in range(n_component):
        # Construct the problem.
        alpha = cp.Variable(n)
        K2 = K**2
        objective = cp.Maximize(cp.quad_form(alpha, K2))
        constraints = [alpha.T @ K @ alpha - 1 == 0]
        if idx >= 1:
            for j in range(idx):
                constraints.append(alpha.T @ K @ components[j] == 0)
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        components.append(alpha)
    return components

