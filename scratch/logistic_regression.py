import numpy as np


class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.max_iter):
            linear_output = np.dot(X, self.weights) + self.bias
            probs = self._sigmoid(linear_output)

            gradients_w = np.dot(X.T, (probs - y)) / X.shape[0]
            gradients_b = np.sum(probs - y) / X.shape[0]

            self.weights -= self.learning_rate * gradients_w
            self.bias -= self.learning_rate * gradients_b

        return self

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        probs = self._sigmoid(linear_output)
        return np.column_stack((1 - probs, probs))
