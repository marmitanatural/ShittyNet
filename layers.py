from numpy import number

import numpy as np
from neuron import Neuron

class Input:
    def __init__(self, layer_size):
        self.layer_size=layer_size
        
    def compute(self, input_data):
        return input_data

class FullyConnected:
    def __init__(self, layer_size, previous_layer_size, activation_function):
        self.layer_size = layer_size 
        self.previous_layer_size = previous_layer_size
        self.neurons = []
        
        for i in range(layer_size):
            self.neurons.append(
                Neuron(
                    activation_function=activation_function,
                    weights=np.random.rand(previous_layer_size),
                    bias=np.random.rand(1)
                )   
            )

    def compute(self,input_data):
        layer_output = np.empty(self.layer_size)

        for i in range(self.layer_size):
            layer_output[i] = self.neurons[i].compute(
                input_data=input_data
            )
                
        return layer_output
    