# Handwritten-Digit-Classifier


## Overview:
----------

* In this project, *Handwritten Digit Classifier*, we will try build a neural network from scratch by using the required algorithms and mathematical functions in order to classify handwritten digit.

* The dataset is provided by *MNIST(Modified National Institute of Standards and Technology)*. MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning.

* The various neural network architectures created in this project are insprised from Michael Nielsen's [Neural Networks and Deep Learning book](http://neuralnetworksanddeeplearning.com/index.html).


## Neural Network's Architecture:
------

* We all know that a basic neural network has three parts:-
    
    1. *an i/p layer*: The leftmost layer in the network is called the *input layer*, and the neurons within the layer are called *input neurons*.
    
    2. *hidden layers*: Can be one or more than one in number. It is the middle layer, and all the computation or processing in the neural network is done by the hidden layers.
    
    3. *an o/p layer*: The rightmost layer in the network is called the *output layer*, and the neurons within the layer are called *output neurons*.
    

  ![basic_nn](https://user-images.githubusercontent.com/33928040/68084021-0a012300-fe56-11e9-8633-022a76a808e8.png)
    
    **Fig 1.1:-** *The above figure shows how a basic neural network looks like. The network has one input layer, one hidden layer and one output layer.*


* So, as we know how a basic neural network looks like, let's now see how our neural network architecture for classifying handwritten digits looks like:-
  
  * To recognize individual digits we will be using a **three-layer** neural network.
  
  * The **input layer** of the network contains neurons encoding the values of the input pixels. Our training data will consist of **28 by 28 pixel images** of scanned handwritten digits, so the input layer contains **784 = 28x28 neurons**.
  
  * The second layer of the netwrok is a **hidden layer**. We denote the number of neurons in this hidden layer by *n*, and we'll experiment with different values of n.
  
  * The **output layer** of the network contains **10 neurons**, having representing values from 0 to 9.
  
  **NOTE:** *The input pixels are greyscales, with value 0.0 for white and 1.0 for black, and in between values representing gradually darkening shades of grey.*
