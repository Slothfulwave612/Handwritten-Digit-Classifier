"""
network2.py
-----------

The neural network implementation here is the improved version of the neural network impemented in network.py file.

Improvements, here includes the addition of the cross-entropy cost function, regularization and better initialization of network weights.

The ``network2.py`` module includes three classes and four functions.

The first class is named as ``QuadraticCost`` which contains methods for our quadratic-cost function, the second class is named as ``CrossEntropyCost`` which contains methods for our cross-entropy cost function and the third class is named as ``Network`` which contains the new implementation of the neural network.

The other four functions are the basic functions, one to load the neural network from the ``filename``, the function will return an instance of the class ``Network``. The second function will return a 10-dimensional unit vector. The third function will compute the sigmoid funtion and the fourth function will compute the derivative of the sigmoid function.
"""

## Libraries
"""
For our module network2.py we will be using json, random, sys and numpy libraries.
"""

import json
import random
import sys
import numpy as np

## Defining the cross-entropy cost function
class CrossEntropyCost:

    @staticmethod
    def fn(a, y):
        """
        Returns the cost associated with an output ``a`` and desired output ``y``(using the cross-entropy formula).

        Note that np.nan_to_num is used to ensire numerical stability. In particular, if both ``a`` and ``y`` 
        have a 1.0 in the same slot, then the expression ``(1-y)*np.log(1-a)`` returns nan. 
        The np.nan_to_num ensures that the value is converted to the correct value ``0.0``.
        """
        return np.nan_to_num( -(y * np.log(a)) - ((1-y) * np.log(1-a)) )

    def delta(self, a, y):
        """
        Return the error delta from the output layer. Note that the parameter ``z`` is not used by the method. 
        It is included in the method's parameters in order to make the interface consistent with the delta method for other cost classes.
        """
        return (a-y)

## Network Class
class Network:

    def __init__(self, sizes):
        """
        The list ``sizes`` contains the number of neurons in
        the respective layers of the network.

        e.g. if the list is [2, 3, 1] then it would be a three-layer network,
        with the first layer containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.

        The biases and weights for the network are initialized randomly, using ``self.default_weight_initializer``.
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.default_weight_initializer()
        self.cost = CrossEntropyCost

    def default_weight_initializer(self):
        """
        Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """

        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        ## initializing the biases

        self.weights = [np.random.randn(y,x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        ## initializing the weights

    def feedforward(self, a):
        """
        Return the output of the network if ``a`` is input.
        """
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda,
            evaluation_data=None,
            monitor_training_accuracy=False,
            monitor_training_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_evaluation_cost=False):

        """
        Train the neural network using mini-batch stochastic gradient descent. 
        The ``training_data`` is a list of tuples ``(x,y)`` representing the training inputs 
        and the desired outputs. The parameter ``lmbda`` is assigned a value ``0.0``, by default.

        The method also accepts ``evaluation_data``, usually either the validation or test data. 
        We can monitor the cost and accuracy on either the evaluation data or the training data, by setting the appropriate flags.

        The method returns a tuple containing four lists:
        the (per-epoch) costs on the evaluation data, the accuracies on the evaluation data, 
        the costs on the training data, and the acuracies on the training data. 
        All values are evaluated at the end of each training epoch.

        e.g. if we train for 30 epochs, then the first element of the tuple will 
        be a 30-element list containing the cost on the evaluation data at the end of each epoch. 
        Note that the lists are empty if the corresponding flag is not set.
        """

        training_data = list(training_data)
        ## since training_data is a zip object
        ## in-order to access the data inside it we have to convert it into the list format

        n_train = len(training_data)
        ## length of training_data

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            ## same as training_data, evaluation_data is a zip object
            ##converting it into list format

            n_eval = len(evaluation_data)
            ## length of evaluation_data

        training_accuracy, training_cost = [], []
        evaluation_accuracy, evaluation_cost = [], []
        ## initializing an empty lists for costs and accuracy for evaluation and training data

        for i in range(epochs):
            random.shuffle(training_data)
            ## shuffling prevents bias during training

            mini_batches = [training_data[k: k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            ## making the mini-batches as per mini-batch-size

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n_train)
                ## calling the function update_mini_batch
                ## the function will update the values for weights and biases

            print(f'Epochs {i+1} training complete')

            if monitor_training_accuracy:
                ## calculation accuracy on training data
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f'Accuracy on training data: {(accuracy/n_train)*100}')

            if monitor_training_cost:
                ## calculation cost on training data
                cost = self.total_cost(training_data, lmbda, convert=False)
                training_cost.append(cost)

            if monitor_evaluation_accuracy:
                ## calculation accuracy on evaluation data
                accuracy = self.accuracy(evaluation_data, convert=False)
                evaluation_accuracy.append(accuracy)
                print(f'Accuracy on evaluation data: {(accuracy/n_eval)*100}')

            if monitor_evaluation_cost:
                ## calculation cost on evaluation data
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)

            print()

        return training_accuracy, training_cost, evaluation_accuracy, evaluation_cost

    def update_mini_batch(self, mini_batch, eta, lmbda, n_train):
        """
        Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
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

        self.weights = [(1-eta*(lmbda/n_train)) * w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        ## updating the values of weights by using gradient descent

        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        ## updating the values of biases by using gradient descent

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """

        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        ## initializing the biases and weights value

        ## feedforward operation:
        ## ----------------------
        activation = x

        activations = [x]
        ## list to store all the activations layer by layer

        zs = []
        ## list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,activation) + b
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

        delta = (self.cost).delta(zs[-1], activations[-1], y)
        ## using delta method from CrossEntropyCost
        ## to compute error in the last layer

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

    def accuracy(self, data, convert):
        """
        Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data.
        """

        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]

        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        ## making a list of tuples ``(x`, y)``
        ## x is the predicted output
        ## y is the correct output

        ## now returning the number of correct predicted outputs
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert):
        """
        Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.
        """

        cost = 0.0
        ## taking initial cost as 0.0

        for x, y in data:
            a = self.feedforward(x)
            ## calling feedforward method

            if convert:
                y = vectorized_result(y)
            ## if test_data or validation_data is passed
            ## converting the output to 10-dimensional unit vector

            cost += self.cost.fn(a,y) / len(data)
            ## calculating the cost
            ## cross-entropy cost function

            cost += 0.5 * (lmbda/len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
            ## regularizing the cost function
            ## by adding the regularization term

        return cost

    def ouput_accuracy(self, test_data):
        """
        Return the number of inputs in ``test_data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        """

        test_data = list(test_data)
        n_test = len(test_data)

        results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]

        accuracy = sum(int(x == y) for x, y in results)

        print(f'Accuracy on test-data: {(accuracy/n_test) * 100}')

    def save(self, filename):
        """
        Save the neural network to the file ``filename``.
        """

        data = {
                'sizes': self.sizes,
                'weights': [w.tolist() for w in self.weights],
                'biases': [b.tolist() for b in self.biases]
                }
        ## making a list that will contain all the info to be stored in the file

        f = open(filename, 'w')
        json.dump(data, f)
        f.close()
        ## creating the file, ``filename`` and adding the info to it in json format

## Functions:
## ----------

def vectorized_result(y):
    """
    Return a 10-dimensional unit vector with a 1.0 in the y'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """

    e = np.zeros((10,1))
    ## making an array of length 10 having zero in all the indices

    e[y] = 1.0
    ## at index y assigning the value 1.0

    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

## slothfulwave612...
