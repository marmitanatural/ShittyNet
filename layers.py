import numpy as np
from layers import Neuron

class Input:
    def __init__(self, input_length):
        self.input_length
    
    def compute(self, inputs):
        return self.inputs 

class FullyConnected:
    def __init__(self, output_size, input_size, activation)
        self.output_size = output_size 
        self.neurons = np.empty(self.output_size)
        self.weights = np.random.rand(self.input_size)
        self.biases = np.random.rand(self.input_size)
        
        for i in range(self.output_size):
            self.neurons[i] = Neuron(
                activation=activation,
                weights=self.weights,
                biases=self.biases
            )

    def compute(self, inputs):
        output = np.empty(self.output_size)

        for i in range(self.output_size):
            output[i] = neurons[i].compute(inputs=inputs)

        return output 

class Output:
    def __init__(self, input_size, weights, biases, activation)
        self.weights = np.random.rand(self.input_size)
        self.biases = np.random.rand(self.input_size)
        self.output_neuron = Neuron(
            activation=activation,
            weights=weights,
            biases=biases
        )

    def compute(self,inputs):
        return self.output_neuron.compute(inputs=inputs)      
        
        