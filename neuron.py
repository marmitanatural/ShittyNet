import numpy as np 

class Neuron:
    def __init__(self, activation_function, weights, bias):
        self.activation_function=activation_function
        self.weights = weights
        self.bias = bias

    def compute(self, input_data):
        return self.activation_function(np.add(np.dot(self.weights, input_data), self.bias))