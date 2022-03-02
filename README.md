# ShittyNet :poop:
In the wise words of George Hotz:
<center>"this may not be the best deep learning framework, but it is a deep learning framework".</center>

## Introduction
ShittyNet is a from scratch implementation of a Neural Network using only numpy and python's default data structures. It was created by the author to gain a beyond conceptual understanding of how a neural network learns and why this might be a challenging to implement in code. Indeed, this is entirely useless for practical purposes but was an interesting exercise.

It can only learn for single output regression tasks (there are no classification loss functions in the package, nor has the case of multiple outputs been implemented). It uses the most vanilla and traditional flavour of gradient descent to learn, with backpropagation being used to compute the gradient. It updates the gradient after every training example has been fed through the network.

## Sample Usage
    from ShittyNet.network import Network
    from ShittyNet.loss_functions import MSE
    from ShittyNet.activation_functions import Linear, ReLU

    training_data = np.random.rand(5000,5)
    labels = np.random.rand(5000,1)

    nn = Network()

    nn.add_input_layer(layer_size=5)
    nn.add_fully_connected_layer(layer_size=4, activation_function=ReLU())
    nn.add_fully_connected_layer(layer_size=2, activation_function=ReLU())
    nn.add_fully_connected_layer(layer_size=1, activation_function=Linear())

    nn.set_training_parameters(epochs=2, learning_rate=0.01, loss_function=MSE())
    nn.train(training_data=training_data, labels=labels)

## To Do
- See if the training actually works by checking if it converges on some common dataset (after all the index gymnastics in this draft of ShittyNet, there are bound to be some bugs).
- Add a training progress logging mechanism.
- Add a prediction function to use the trained model to predict.
- Adapt to accept multiple outputs and adapt for a wide variety of tasks by putting in the relevant loss functions.
- Add regularization or early stopping.
- Add batch mechanism.
- Massive refactor to improve performance and readability as currently it is definitely a bit like :smiling_face_with_tear:	
- Add modular type of API like Keras functional API and 'exotic' layers like dropout, normalization, concat, dot, ..
- The list never ends, in effect one could keep going until you copy everything in the Keras repository.

## Disclaimer
Please do not think that this is an industrial strength deep learning library to replace Tensorflow or PyTorch. It is not. It is a little experiment to deepen the author's knowledge of deep learning (beyond a conceptual understanding). It is probably riddled with bugs. It will probably never be actively updated. 