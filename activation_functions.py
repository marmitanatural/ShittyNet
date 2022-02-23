import numpy as np


class Linear:
    def compute(self, x):
        return x

    def derivative_compute(self, x):
        return 1


class ReLU:
    def compute(self, x):
        return 0 if x <= 0 else x

    def derivative_compute(self, x):
        return 0 if x <= 0 else 1


class Sigmoid:
    def compute(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_compute(self, x):
        return np.exp(-x) / ((np.exp(-x) + 1) ** 2)
