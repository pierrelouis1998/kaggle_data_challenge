import numpy as np
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from kaggle_data_challenge.scratch.logistic_regression import SimpleLogisticRegression


class SupportVectorClassification(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.support_vectors_idx = None
        self.alphas = None

    def solve_qp(self, P, q, G, h, A, b):
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        return np.array(sol['x']).flatten()

    def fit(self, K, y):
        n_samples = len(y)
        classes = np.unique(y)
        self.classes = classes
        y = np.where(y == classes[0], -1, 1)
        y = y.astype(float)

        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        G = np.vstack((np.eye(n_samples), -np.eye(n_samples)))
        h = np.hstack((self.C * np.ones(n_samples), np.zeros(n_samples)))
        A = y.reshape(1, -1)
        b = np.zeros(1)

        alphas = self.solve_qp(P, q, G, h, A, b)
        support_vectors_idx = alphas > 1e-6
        self.support_vectors_idx = support_vectors_idx
        self.alphas = alphas[support_vectors_idx]
        self.support_vectors_ = K[support_vectors_idx]
        self.support_vector_labels_ = y[support_vectors_idx]

        self.intercept_ = np.mean(
            self.support_vector_labels_
            - np.sum(
                self.alphas
                * self.support_vector_labels_
                * K[support_vectors_idx][:, support_vectors_idx],
                axis=1,
            )
        )

        decision_values = self.decision_function(K)
        self.probability_estimator_ = SimpleLogisticRegression().fit(decision_values.reshape(-1, 1), y)

        return self

    def decision_function(self, K):
        return np.dot(K, self.alphas * self.support_vector_labels_) + self.intercept_

    def predict(self, X):
        decision_values = self.decision_function(X)
        return np.where(decision_values >= 0, self.classes[1], self.classes[0])

    def predict_proba(self, K):
        decision_values = self.decision_function(K)
        return self.probability_estimator_.predict_proba(decision_values.reshape(-1, 1))
