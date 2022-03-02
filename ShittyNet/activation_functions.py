import numpy as np


class Linear:
    def compute(self, x):
        return x

    def derivative_compute(self, x):
        return np.vectorize(lambda z: 1)(x)


class ReLU:
    def compute(self, x):
        return 0 if x <= 0 else x

    def derivative_compute(self, x):
        return np.vectorize(lambda z: 0 if z <= 0 else 1)(x)


class Sigmoid:
    def compute(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_compute(self, x):
        return np.vectorize(lambda z: np.exp(-z) / ((np.exp(-z) + 1) ** 2))(x)
