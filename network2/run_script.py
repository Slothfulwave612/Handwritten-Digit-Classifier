"""
run_script.py
-------------

The module will let the users run our neural network.

The program will return the accuracy after each epochs
"""

import mnist_loader
## importing the mnist_loader module
## used for loading the mnist dataset

import network2
## network1 is the module that implements the neural network concepts

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
## using the load_data_wrapper() to prepare the dataset in the required format

net = network2.Network([784, 30, 30, 10])
## constructing the neural network
## having 784 i/p neurons,
## 30 neurons in the hidden layer(n/w has only one hidden layer)
## and 10 o/p neurons representing 0...9 digits

training_accuracy, training_cost, evaluation_accuracy, evaluation_cost = net.SGD(training_data, epochs=30, mini_batch_size=10, eta=0.1,
    lmbda = 5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_evaluation_cost=False,
    monitor_training_accuracy=True,
    monitor_training_cost=False)
## calling SGD() function to perform the mini-batch gradient descent
## the function will print the accuracy of each epochs

net.ouput_accuracy(test_data)

net.save('network2')

## slothfulwave612...
