"""
network1.py
-----------

A module to implement the mini-batch gradient descent learning algorithm
for a feedforward neural network. Gradients are calculated using backpropagation.

The network1.py module includes a class and three functions.

The class is named as ``Network`` which contains methods for computing
the gradient by performing the backpropogation and calculating
the accuracy by using the trained weights and biases.

The other three miscellaneous functions are the basic functions, one to calculate
the sigmoid function, second to calculate the
sigmoid prime function(i.e. partial derivative of the sigmoid function)
and third to calculate the cost derivative(partial derivative of the cost function).
"""

## Libraries
"""
For our network1.py module we will be using the numpy library for all the
numerical computation and random module for making random results.
"""
import numpy as np
import random

class Network:

    def __init__(self, sizes):
        """
        The list ``sizes`` contains the number of neurons in
        the respective layers of the network.

        e.g. if the list is [2, 3, 1] then it would be a three-layer network,
        with the first layer containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.

        The biases and weights for the network are initialized randomly,
        using a Guassian distribution with mean 0 and variance 1.

        Note that the first layer is the input layer, and by convention
        we won't set any biases for those neurons, since biases are
        only ever used in computing the outputs from later layers.
        """

        self.num_layers = len(sizes)
        ## total number of layers in the network

        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        ## randomly initializing the values for biases

        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        ## randomly initializing the values for weights

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, test_data, epochs, mini_batch_size, eta):
        """
        Training the neural network using mini-batch stochastic gradient descent.

        The ``training_data`` is a list of tuples ``(x,y)``
        representing the training inputs and the desired outputs.

        The ``test_data``, same as the ``training_data``, is a list of tuples
        ``(x,y)`` representing the training inputs and
        the desired outputs. It is there so that we can test
        the accuracy of our network.

        The parameter ``epochs`` refers to the number of times
        the learning algorithm will walk through the entire training dataset.

        ``mini_batch_size`` is the batch size.

        ``eta`` is the learning rate.
        """

        test_data = list(test_data)
        ## since test_data is a zip object
        ## in-order to access the data inside it we have to convert it into the list format

        training_data = list(training_data)
        ## same as test_data, training_data is a zip object
        ## converting it into list format

        n_test, n_train = len(test_data), len(training_data)
        ## length of test_data and train_data

        ## Mini-Batch Stochastic Gradient Descent:
        for j in range(epochs):
            random.shuffle(training_data)
            ## shuffling prevents the any bias during training

            mini_batches = [training_data[k: k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            ## making the mini-batches as per mini-batch-size

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            ## calling the function update_mini_batch(see below)
            ## the funtion will update the values for weights and biases

            print(f'Epoch {j}: {self.evaluate(test_data)}/{n_test}')
            ## finding the accuracy after each epoch
            ## evaluate() function is called(see below)

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini-batch.

        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta`` is the learning rate.
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        ## nabla_b is the list that will store change in the values of biases
        ## every bias's value is initially initialized as 0

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        ## nabla_w is the list that will store change in the values of weights
        ## every weight's value is initially initialized as 0

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            ## calling backprop() function(see below)
            ## delta_nabla_b and delta_nabla_w will contain change in weights and biases respectively

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            ## updating the values in nabla_b list

            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            ## updating the values in nabla_w list

        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        ## updating the values of weights by using gradient descent

        self.baises = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        ## updating the values of biases by using gradient descent

    def backprop(self, x, y):
        """
        Return a tuple ``(delta_nabla_b, delta_nabla_w)`` representing the gradient for the cost function C_x.

        ``delta_nabla_b`` and ``delta_nabla_w`` are layer-by-layer lists of numpy arrays, similar to ``self.biases`` and ``self.weights``.
        """

        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        ## initializing the baises and weights value

        ## feedforward operation:
        ## ----------------------
        activation = x

        activations = [x]
        ## list to store all the activations layer by layer

        zs = []
        ## list to store all the z vectors, layer by layer

        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            ## calculating the z value, as per the formula

            zs.append(z)
            ## appending the result to zs array

            activation = sigmoid(z)
            ## calling sigmoid function(see below)
            ## to perform sigmoid operation, as per the formula

            activations.append(activation)
            ## appending the result to activations array

        ## backward pass:
        ## --------------

        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        ## calculating the error in the last layer
        ## calling cost_derivative and sigmoid_prime functions(see below)

        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        ## using equations of backpropagation
        ## to assign the change in values of weights and biases for the final layer

        ## now calculating the errors for the rest of the layer
        ## and calculating the change in weights and biases for each layer
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            ## assigning the z values of particular layers

            sp = sigmoid_prime(z)
            ## calculating the sigmoid_prime for z values

            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            ## calculating errors for rest of the layers

            delta_nabla_b[-layer] = delta
            delta_nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
            ## assigning the change in values of biases and weights for rest of the layers

        return (delta_nabla_b, delta_nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network outputs the correct result. Note that the neural network's output is assumed to be the index of whicever neuron in the final layer has the highest activation.
        """

        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        ## making a list of tuples ``(x`, y)``
        ## x` is the predicted output
        ## y is the correct output

        ## now returning the number of correct predicted outputs
        return sum(int(x == y) for (x,y) in test_results)

### Miscellaneous Functions:
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

def cost_derivative(output_activations, y):
    """Derivative of cost function."""
    return (output_activations - y)

### slothfulwave612...
