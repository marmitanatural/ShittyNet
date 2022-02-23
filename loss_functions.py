class MSE:
    def compute(self, prediction, target):
        return (1 / 2.0) * (target - prediction) ** 2

    def derivative_compute(self, prediction, target):
        return target - prediction
