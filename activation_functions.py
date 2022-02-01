import numpy as np


class Linear:
    def compute(self, x):
        return x


class ReLU:
    def compute(self, x):
        return 0 if x <= 0 else x


class Sigmoid:
    def compute(self, x):
        return 1 / (1 + np.exp(-x))
