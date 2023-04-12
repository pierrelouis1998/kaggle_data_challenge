import numpy as np
from cvxopt import matrix, spdiag
from cvxopt.coneprog import qp
from sklearn.svm import SVC
from tqdm import tqdm


class RMKL:
    def __init__(self, C=1.0, lr=1e-2, max_iter=1000):
        self.R: np.ndarray
        self.Y: np.ndarray
        self.weights: np.ndarray
        self.objective: np.ndarray
        self.ker_matrix: np.ndarray
        self.bias: np.ndarray
        self.dual_coef: np.ndarray

        self.C = C
        self.learning_rate = lr
        self.max_iter = max_iter

    def _prepare(self, KL, Y):
        '''preprocess data before training'''
        self.KL, self.Y = KL, Y
        self.n_kernels = len(self.KL)

        self.classes_ = np.unique(self.Y)

        return

    def fit(self, X, y):
        self._prepare(X, y)
        self.combine_kernels()

    def initialize_optimization(self):
        self.R = np.array([self.compute_radius(K) ** 2 for K in self.KL])
        self.Y = np.array([1 if y == self.classes_[1] else -1 for y in self.Y])

        weights = np.ones(self.n_kernels) / self.n_kernels
        ker_matrix = self.sum_kernels(self.KL, weights)

        mar, gamma = self._get_gamma(ker_matrix, self.Y)
        yg = gamma.T * self.Y
        bias = 0.5 * (np.dot(gamma, np.dot(ker_matrix, yg)))

        coef_r = np.dot(self.R, weights) / self.C
        obj = np.dot(yg, np.dot((ker_matrix + np.eye(len(gamma)) * coef_r), yg)) + gamma.sum() + 0

        self.weights = weights
        self.objective = obj
        self.ker_matrix = ker_matrix
        self.bias = bias
        self.dual_coef = gamma

    def combine_kernels(self):
        self.initialize_optimization()

        for _ in tqdm(range(self.max_iter)):
            self.do_step()

    def do_step(self):
        Y = self.Y

        # weights update
        yg = self.dual_coef.T * Y

        grad = np.array([np.dot(yg, np.dot((K + rk / self.C * np.eye(len(yg))), yg)) \
                         for rk, K in zip(self.R, self.KL)])
        weights = self.weights + self.learning_rate * grad
        weights[weights < 0] = 0
        weights /= sum(weights)

        # compute combined kernel
        ker_matrix = self.sum_kernels(self.KL, weights)

        # margin (and gamma) update
        mar, gamma = self._get_gamma(ker_matrix, self.Y)

        # compute objective and bias
        yg = gamma * Y
        coef_r = np.dot(self.R, weights) / self.C
        obj = np.dot(yg, np.dot((ker_matrix + np.eye(len(gamma)) * coef_r), yg)) + gamma.sum() + 0
        bias = 0.5 * np.dot(gamma, np.dot(ker_matrix, yg))

        self.weights = weights
        self.objective = obj
        self.ker_matrix = ker_matrix
        self.dual_coef = gamma
        self.bias = bias

    def _get_gamma(self, K, Y):
        svm = SVC(C=self.C, kernel='precomputed').fit(K, Y)
        n = len(Y)
        gamma = np.zeros(n)
        gamma[svm.support_] = np.array(svm.dual_coef_)
        idx_pos = gamma > 0
        idx_neg = gamma < 0
        sum_pos, sum_neg = gamma[idx_pos].sum(), gamma[idx_neg].sum()
        gamma[idx_pos] /= sum_pos
        gamma[idx_neg] /= sum_neg
        gammay = gamma * Y
        obj = np.dot(gammay, np.dot(K, gammay)) ** .5
        return obj, gamma

    def predict(self, KL):
        return np.array([self.classes_[1] if p >= 0 else self.classes_[0] for p in self.score(KL)])

    def score(self, KL):
        Kte = self.sum_kernels(KL, self.weights)
        ygamma = self.dual_coef.T * np.array([1 if y == self.classes_[1] else -1 for y in self.Y])
        return Kte @ ygamma - self.bias

    def sum_kernels(self, KL, weights):
        return (KL * weights[:, None, None]).sum(axis=0)

    def compute_radius(self, K):
        K = K.astype(np.double)
        n = K.shape[0]

        P = 2 * matrix(K)
        p = -matrix(K.diagonal())
        G = -spdiag([1.0] * n)
        h = matrix([0.0] * n)
        A = matrix([1.0] * n).T
        b = matrix([1.0])

        objective = qp(P, p, G, h, A, b)['primal objective']

        return abs(objective) ** .5
