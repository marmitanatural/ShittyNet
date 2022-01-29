import numpy as np 

class Neuron:
    def __init__(self, activation, weights, biases):
        self.activation=activation
        self.weights = weights
        self.biases = biases

    def compute(self, inputs):
        return self.activation(np.add(np.multiply(self.weights, inputs), self.biases))