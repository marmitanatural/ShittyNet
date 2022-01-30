from layers import Input, FullyConnected


class Network():
    def __init__(self):
        self.layers=[] 
        self.layer_indexer=None

    def add_input_layer(self, layer_size):
        self.layers.append(
            Input(
                layer_size=layer_size
            )
        )
        self.layer_indexer=0

    def add_fully_connected_layer(self, layer_size, activation_function):
        if self.layer_indexer is None:
            return "Please use an input layer as the first layer of the network."

        self.layers.append(
            FullyConnected(
                layer_size=layer_size,
                previous_layer_size=getattr(self.layers[self.layer_indexer], "layer_size"), 
                activation_function=activation_function
            )
        )

        self.layer_indexer+=1

    def set_training_parameters(self, epochs, loss_function):
        self.epochs=epochs
        self.loss_function=loss_function
    
    def feed_forward(self, input_data):
        pass

"""
nn = Network()

nn.add_input_layer(layer_size=5)
nn.add_fully_connected_layer(layer_size=4, activation_function=lambda x: 5*x)
nn.add_fully_connected_layer(layer_size=2, activation_function=lambda x: 2*x)
nn.add_fully_connected_layer(layer_size=1, activation_function=lambda x: x)
nn.set_training_parameters(epochs=500, loss_function="squared")

input_data -> network -> prediction -> compute loss -> back_propagation
 (not sure if this is done for every training example or for batches or per epoch)
"""