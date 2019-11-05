"""
run_script.py
-------------

The module will let the users run our neural network.

The program will return the accuracy after each epochs
"""

import mnist_loader
## importing the mnist_loader module
## used for loading the mnist dataset

import network1
## network1 is the module that implements the neural network concepts

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
## using the load_data_wrapper() to prepare the dataset in the required format

net = network1.Network([784, 30, 10])
## constructing the neural network
## having 784 i/p neurons,
## 30 neurons in the hidden layer(n/w has only one hidden layer)
## and 10 o/p neurons representing 0...9 digits

net.SGD(training_data, test_data, epochs=30, mini_batch_size=10, eta=3.0)
## calling SGD() function to perform the mini-batch gradient descent
## the function will print the accuracy of each epochs

## slothfulwave612...
