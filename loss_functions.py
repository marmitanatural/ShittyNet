import numpy as np


class MAE:
    def compute(self, prediction, target):
        return np.abs(target - prediction)


class MSE:
    def compute(self, prediction, target):
        return (target - prediction) ** 2


class RMSE:
    def compute(self, prediction, target):
        return np.sqrt((target - prediction) ** 2)
