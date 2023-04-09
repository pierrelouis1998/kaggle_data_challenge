import numpy as np
import cvxpy as cp


def kernel_svm(K: np.ndarray, labels: np.array, lbda: float):
    """Perform kernel SVM"""
    n = K.shape[0]
    alpha = cp.Variable(n)
    lamb = cp.Parameter(nonneg=True)
    lamb.value = lbda
    Kcv = cp.Parameter(shape=(n,n), value=K, PSD=True)
    objective = cp.Maximize((2*alpha.T @ labels) - cp.QuadForm(alpha, Kcv))
    constraints = [cp.multiply(labels, alpha) >= 0, cp.multiply(labels, alpha) <= 1 / (1 * lamb * n)]
    problem = cp.Problem(objective, constraints)
    results = problem.solve()
    return alpha.value
