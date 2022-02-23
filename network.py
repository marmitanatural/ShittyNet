from layers import Input, FullyConnected
from activation_functions import Linear, ReLU
from loss_functions import MSE
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.input_only = True
        self.layer_indexer = 0
        self.unactivated_outputs = []  # [z1, z2, ...zl]
        self.activated_outputs = []  # [a1, a2, ...al]
        self.output_errors_reversed = []  # [delta_l, delta_l-1, ... delta_1]

    def add_input_layer(self, layer_size):
        self.input_layer_size = layer_size

    def add_fully_connected_layer(self, layer_size, activation_function):
        if self.input_only is True:
            self.layers.append(
                FullyConnected(
                    layer_size=layer_size,
                    previous_layer_size=self.input_layer_size,
                    activation_function=activation_function,
                )
            )
            self.input_only = False
        else:
            self.layers.append(
                FullyConnected(
                    layer_size=layer_size,
                    previous_layer_size=getattr(
                        self.layers[self.layer_indexer - 1], "layer_size"
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
        for layer_index in range(self.layer_indexer + 1):
            unactivated_output, activated_output = self.layers[layer_index].compute(
                pass_forward
            )
            self.unactivated_outputs.append(unactivated_output)
            self.activated_outputs.append(activated_output)

            pass_forward = activated_output

        return pass_forward

    def compute_output_error(self, label, prediction):
        self.output_errors_reversed.append(
            self.loss_function.derivative_compute(label, prediction)
            * getattr(self.layers[-1], "activation_function").derivative_compute(
                self.unactivated_outputs[-1]
            )
        )

    def backpropagate_error(self):
        for index in range(self.layer_indexer):
            self.output_errors_reversed.append(
                np.multiply(
                    np.matmul(
                        self.layers[self.layer_indexer - index].weights,
                        self.output_errors_reversed[index],
                    ),
                    self.layers[
                        self.layer_indexer - index - 1
                    ].activation_function.derivative_compute(
                        self.unactivated_outputs[self.layer_indexer - index - 1]
                    ),
                )
            )

    def compute_gradient(self):
        pass

    def train(self, training_data, labels):
        for item, label in zip(training_data, labels):
            prediction = self.feed_forward(item)
            self.compute_output_error(label, prediction)
            self.backpropagate_error()


nn = Network()

nn.add_input_layer(layer_size=2)
nn.add_fully_connected_layer(layer_size=2, activation_function=ReLU())
nn.add_fully_connected_layer(layer_size=1, activation_function=Linear())
nn.set_training_parameters(epochs=100, loss_function=MSE())
nn.train(training_data=np.array([[0.5, 0.4]]), labels=np.array([0.7, 0.8]))
"""
input_data -> network -> prediction -> compute loss -> back_propagation
 (not sure if this is done for every training example or for batches or
 per epoch)
"""
