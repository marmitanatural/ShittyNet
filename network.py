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
        pass_forward = input_data
        for layer_index in range(1, self.layer_indexer + 1):
            pass_forward = self.layers[layer_index].compute(pass_forward)

        return pass_forward

    def train(self, training_data, labels):
        for item, label in zip(training_data, labels):
            prediction = self.feed_forward(item)
            loss = self.loss_function.compute(prediction, label)

        return loss
        # do backprop to compute gradient and update weights


nn = Network()

nn.add_input_layer(layer_size=3)
nn.add_fully_connected_layer(layer_size=2, activation_function=ReLU())
nn.add_fully_connected_layer(layer_size=1, activation_function=Linear())
nn.set_training_parameters(epochs=100, loss_function=MSE())
nn.train(training_data=np.array([[0.45, 0.9, 0.3]]), labels=np.array([0.7]))
"""
input_data -> network -> prediction -> compute loss -> back_propagation
 (not sure if this is done for every training example or for batches or
 per epoch)
"""
