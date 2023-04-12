import numpy as np
from cvxopt import matrix, spdiag, solvers
from sklearn.svm import SVC
from tqdm import tqdm

from kaggle_data_challenge.scratch.svc import SupportVectorClassification


class RMKL:
    def __init__(self, C=1.0, lr=1e-2, max_iter=1000, solver="libsvm"):
        self.kernel_radius: np.ndarray
        self.y: np.ndarray
        self.weights: np.ndarray
        self.kernel_matrix: np.ndarray
        self.gamma: np.ndarray

        self.C = C
        self.learning_rate = lr
        self.num_iter = max_iter
        self.solver = solver

    def fit(self, kernel_list, y):
        self.kernel_list, self.y = kernel_list, y
        self.n_kernels = len(self.kernel_list)
        self.classes = np.unique(self.y)

        self.combine_kernels()

    def initialize_optimization(self):
        self.kernel_radius = np.array([self.compute_radius(K) ** 2 for K in self.kernel_list])
        self.y = np.array([1 if y == self.classes[1] else -1 for y in self.y])

        self.weights = np.ones(self.n_kernels) / self.n_kernels
        self.kernel_matrix = self.sum_kernels(self.kernel_list, self.weights)

        self.gamma = self.compute_gamma()

    def combine_kernels(self):
        self.initialize_optimization()

        for _ in tqdm(range(self.num_iter)):
            self.optimize()

    def optimize(self):
        yg = self.gamma * self.y

        grad = np.array([np.dot(yg, np.dot((K + rk / self.C * np.eye(len(yg))), yg)) \
                         for rk, K in zip(self.kernel_radius, self.kernel_list)])
        weights = self.weights + self.learning_rate * grad
        weights[weights < 0] = 0
        weights /= sum(weights)

        self.weights = weights
        self.kernel_matrix = self.sum_kernels(self.kernel_list, weights)
        self.gamma = self.compute_gamma()

    def compute_gamma(self):
        n = len(self.y)

        if self.solver == "libsvm":
            svm = SVC(C=self.C, kernel='precomputed').fit(self.kernel_matrix, self.y)
            gamma = np.zeros(n)
            gamma[svm.support_] = np.array(svm.dual_coef_)
        else:
            svm = SupportVectorClassification(C=self.C)
            svm.fit(self.kernel_matrix, self.y)

            gamma = np.zeros(n)
            gamma[svm.support_vectors_idx] = np.array(svm.alphas)
            gamma *= self.y
        idx_pos = gamma > 0
        idx_neg = gamma < 0
        sum_pos, sum_neg = gamma[idx_pos].sum(), gamma[idx_neg].sum()
        gamma[idx_pos] /= sum_pos
        gamma[idx_neg] /= sum_neg

        return gamma

    def predict(self, kernel_list):
        return np.array([self.classes[1] if p >= 0 else self.classes[0] for p in self.score(kernel_list)])

    def score(self, kernel_list):
        kernel_matrix = self.sum_kernels(kernel_list, self.weights)
        yg = self.gamma * self.y

        bias = 0.5 * np.dot(self.gamma, np.dot(self.kernel_matrix, yg))

        return np.dot(kernel_matrix, yg) - bias

    def sum_kernels(self, kernel_list, weights):
        return (kernel_list * weights[:, None, None]).sum(axis=0)

    def compute_radius(self, K):
        K = K.astype(np.double)
        n = K.shape[0]

        P = 2 * matrix(K)
        p = -matrix(K.diagonal())
        G = -spdiag([1.0] * n)
        h = matrix([0.0] * n)
        A = matrix([1.0] * n).T
        b = matrix([1.0])

        solvers.options['show_progress'] = False
        objective = solvers.qp(P, p, G, h, A, b)['primal objective']

        return abs(objective) ** .5

    def get_kernel_matrix(self):
        return self.sum_kernels(self.kernel_list, self.weights)