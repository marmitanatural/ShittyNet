from layers import Input, FullyConnected
from activation_functions import Linear, ReLU
from loss_functions import MSE
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.layer_indexer = None

    def add_input_layer(self, layer_size):
        self.layers.append(Input(layer_size=layer_size))
        self.layer_indexer = 0

    def add_fully_connected_layer(self, layer_size, activation_function):
        if self.layer_indexer is None:
            return "Use an input layer as the first layer of the network."

        self.layers.append(
            FullyConnected(
                layer_size=layer_size,
                previous_layer_size=getattr(
                    self.layers[self.layer_indexer], "layer_size"
                ),
                activation_function=activation_function,
            )
        )

        self.layer_indexer += 1

    def set_training_parameters(self, epochs, loss_function):
        self.epochs = epochs
        self.loss_function = loss_function

    def feed_forward(self, input_data):
        layer_input = input_data
        for index in range(1, self.layer_indexer + 1):
            layer_output = np.empty(getattr(self.layers[index], "layer_size"))
            for index, neuron in enumerate(getattr(self.layers[index], "neurons")):
                layer_output[index] = neuron.compute(layer_input)
            layer_input = layer_output

        return layer_output

    def train(self, training_data, test_data):
        # training data np.arr(np.arr, ....)
        for item, label in zip(training_data, test_data):
            prediction = self.feed_forward(item)
            loss = self.loss_function.compute(np.array(label), prediction)

            # backprop to update weights
        pass


nn = Network()

nn.add_input_layer(layer_size=7)
nn.add_fully_connected_layer(layer_size=5, activation_function=ReLU())
nn.add_fully_connected_layer(layer_size=5, activation_function=ReLU())
nn.add_fully_connected_layer(layer_size=1, activation_function=Linear())
nn.set_training_parameters(epochs=100, loss_function=MSE())
print(nn.feed_forward(np.array([1, 2, 3, 1, 2, 3, 1])))

"""
input_data -> network -> prediction -> compute loss -> back_propagation
 (not sure if this is done for every training example or for batches or
 per epoch)
"""
