## Artificial Neurons
The most fundamental unit of the neural network is the artificial neuron (inspired by the behavior of biological neurons). Mathematically, it can be represented by:
    $$ y = \sigma(w_i*x_i + b) $$
where $y$ is the output, $\sigma$ is the activation function, $w_i$ is a vector of weights, $x_i$ are the inputs ($x_i$ and $w_i$ are the same length, in fact $x_i$ determines the number of weights), $b$ is the bias.

## Neural Networks
Are stacks of layers of artificial neurons. There are 2 priviledged layers. 
1. The input layer: it does no processing of the incoming data (so no weights, biases or activation function), it simply 'fans' the data out and feeds it to the next layer.
2. The output layer: for regression, generally a single neuron. For classification, 2 or more. It generally has a different activation function than the hidden layers.
The intermediate layers, or 'hidden' layers generally take the same form for simple feed forward networks. In such a setting, they are fully connected to the previous layer and fully connected to the following one. 

TO DO:
    Verify compute function for neuron and for layer
    Make feed forward function for entire network
    Make a compute loss function for the network 
    Figure out how to use backprop to update the weights and biases
    Package these into a training function
