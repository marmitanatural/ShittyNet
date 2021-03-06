from layers import FullyConnected
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.input_only = True
        self.layer_indexer = 0
        self.z = []  # [z1, z2, ...zl]
        self.a = []  # [a1, a2, ...al]
        self.delta = []  # [delta_l, delta_l-1, ... delta_1] until it gets reversed

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
                    previous_layer_size=getattr(self.layers[-1], "layer_size"),
                    activation_function=activation_function,
                )
            )
            self.layer_indexer += 1

    def set_training_parameters(self, epochs, learning_rate, loss_function):
        self.epochs = epochs
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def feed_forward(self, input_data):
        pass_forward = input_data
        for layer_index in range(self.layer_indexer + 1):
            unactivated_output, activated_output = self.layers[layer_index].compute(
                pass_forward
            )
            self.z.append(unactivated_output)
            self.a.append(activated_output)

            pass_forward = activated_output

        return pass_forward

    def compute_output_error(self, label, prediction):
        self.delta.append(
            self.loss_function.derivative_compute(label, prediction)
            * getattr(self.layers[-1], "activation_function").derivative_compute(
                self.z[-1]
            )
        )

    def backpropagate_error(self):
        for index in range(self.layer_indexer + 1):
            self.delta.append(
                np.multiply(
                    np.matmul(
                        self.layers[self.layer_indexer - index].weights.T,
                        self.delta[index],
                    ),
                    self.layers[
                        self.layer_indexer - index - 1
                    ].activation_function.derivative_compute(
                        self.z[self.layer_indexer - index - 1]
                    ),
                )
            )
        self.delta = self.delta[::-1]

    def compute_gradient_update_weights(self, item):
        for index, layer in enumerate(self.layers):
            gradient_biases = self.delta[index + 1]
            gradient_weights = np.matmul(
                self.a[index - 1].reshape(len(self.a[index - 1]), 1)
                if index != 0
                else item.reshape(len(item), 1),
                self.delta[index + 1].reshape(len(self.delta[index + 1]), 1).T,
            ).T

            layer.weights = np.subtract(
                layer.weights, self.learning_rate * gradient_weights
            )

            layer.biases = np.subtract(
                layer.biases, self.learning_rate * gradient_biases
            )

    def reset(self):
        self.z = []
        self.a = []
        self.delta = []

    def train(self, training_data, labels):
        for epoch_counter in range(self.epochs):
            for item, label in zip(training_data, labels):
                prediction = self.feed_forward(item)
                self.compute_output_error(label, prediction)
                self.backpropagate_error()
                self.compute_gradient_update_weights(item)
                self.reset()
        print("training complete")
