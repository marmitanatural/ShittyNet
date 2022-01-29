import numpy as np 
from layers.py import Input, FullyConnected, Output

class Network:
    def __init__(self):
        self.layers = {"fully_connected":[]}

    def add_input_layer(self, input_length):
        self.layers["input"] =  Input(input_length=input_length)
    
    def add_fully_connected_layer(self, length, activation, weights, biases):
        self.layers["fully_connected"].append(
            FullyConnected(
                length=length, 
                activation=activation,
                weights=weights, 
                biases=biases
            )

    def add_output_layer(self):
        self.layers["output"] = Output(activation=activation, weights=weights, biases=biases)

    def set_network_parameters(self, loss_function, optimizer)
        self.loss_function=loss_function
        self.optimizer=optimizer

    def train(self, epochs):
        self.epochs=epochs 

        for i in range(self.epochs):
            pass


        


"""
model = Network()

model.add_input_layer(inputs=numpy_array)
model.add_fully_connected_layer(output_size=5, input_size=len(numpy_array) activation=relu, weights=random, biases=random)
model.add_output_layer(input_size=1, activation=identity, weights=random, biases=random)

model.set_training_parameters(loss_function=loss, optimizer=adam)

model.train(epochs=500)
"""