import numpy as np


class Input:
    def __init__(self, layer_size):
        self.layer_size = layer_size


class FullyConnected:
    def __init__(self, layer_size, previous_layer_size, activation_function):
        self.layer_size = layer_size
        self.previous_layer_size = previous_layer_size
        self.activation_function = activation_function
        self.augmented_weights = np.random.random((layer_size, previous_layer_size + 1))

    def compute(self, input_data):
        input_data = np.append(input_data, 1)

        raw_output = np.matmul(self.augmented_weights, input_data)

        activated_output = np.array(
            [self.activation_function.compute(x) for x in raw_output]
        )

        return activated_output
