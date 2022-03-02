import numpy as np


class FullyConnected:
    def __init__(self, layer_size, previous_layer_size, activation_function):
        self.layer_size = layer_size
        self.previous_layer_size = previous_layer_size
        self.activation_function = activation_function
        self.weights = np.random.random((layer_size, previous_layer_size))
        self.biases = np.random.random((layer_size))
        self.gradient_weights = np.array([])
        self.gradient_bias = np.array([])

    def compute(self, input_data):
        unactivated_output = np.add(np.matmul(self.weights, input_data), self.biases)

        activated_output = np.array(
            [self.activation_function.compute(x) for x in unactivated_output]
        )

        return unactivated_output, activated_output
