# network1

## Overview:


* Here we will try to build a very basic and simple neural network by using all the simple mathematical functions and algorithms to begin with. After building the neural network we will train it on the MNIST dataset's training data and will test our model on the test data so to see how well our network is performing.

* *network1* directory includes two python files :-
  
  * *netwrok1.py*: This file will contain the python code used for the construction of our neural network.
  
  * *mnist_loader.py*: This file will contain the python code used for loading the MNIST dataset, i.e. loading the train and test dataset, for our network to use.
  
  
## Brief Background:

### Perceptrons:
  
  * A *perceptron* is a function or a neural network unit that takes **several inputs** and produces a **single output**.
  
    ![perceptron](https://user-images.githubusercontent.com/33928040/68084256-5437d380-fe59-11e9-8915-4682335b3460.png)
  
    **Fig 1.1:-** The figure shows the perceptron that has **three inputs**(x1, x2 and x3). 
  
  * Now to compute output, we have what we call **weights**, w1, w2,...., real numbers expressing the importance of the respective inputs to the output. The neuron's output, 0 or 1, is determined by wheter the weighted sum *∑j wjxj* is less than or greater than some *threshold value*.
   
      ![percept](https://user-images.githubusercontent.com/33928040/68084321-361ea300-fe5a-11e9-82ce-c2ca57be2019.PNG)

   * Let's now simplyfy the way we describe perceptron, the first change is to write *∑j wjxj* as a dot product, *w.x = ∑j wjxj*, where w and x are vector whose components are the weights and inputs, respectively. The second change that we can do here is, move the threshold to the other side of the inequality, and replace it by what's known as the **perceptron's bias**, *b = -threshold*.
   
      ![percept-1](https://user-images.githubusercontent.com/33928040/68084387-bf35da00-fe5a-11e9-8963-6321d09dd29b.PNG)
  
   * The **bias** is a measure of how easy it is to get the percptron to output a 1. For a perceptron with really big bias, it's extremely easy for the perceptron to output a 1. But if the bias is very negative, then it's difficult for a percptron to output a 1.
   
### Sigmoid Neuron:

  * In our neural netwrok implementation for building a classifier to classify handwritten digits, we will use a **sigmoid neuron**.
  
  * **Sigmoid Neurons** are similar to perceptrons, but modified so that small change in their weights and biases cause only a small change in their output.
  
  * Just like a perceptron, the sigmoid neuron has inputs, weights and biases. The output of the sigmoid neuron is, **σ(w⋅x+b)**, where σ is called the **sigmoid function**, and is defined by:
  
      ![sigmoid](https://user-images.githubusercontent.com/33928040/68084598-a844b700-fe5d-11e9-8286-ae8b48065bde.PNG)

   * To put it all a little more explicitly, the output of a sigmoid neuron with inputs x1,x2,…, weights w1,w2,…, and bias b is:
                                                        
      ![sigmoid2](https://user-images.githubusercontent.com/33928040/68084604-cf02ed80-fe5d-11e9-8f7c-36765f628c79.PNG)
      
   * The reason for using the sigmoid function instead of the perceptron is:
   
      ![sigmoid_graph](https://user-images.githubusercontent.com/33928040/68084701-22297000-fe5f-11e9-9542-8f7324fa8a7b.PNG)   ![step_graph](https://user-images.githubusercontent.com/33928040/68084705-31a8b900-fe5f-11e9-8370-5a8e18682805.PNG)
      
      if σ has been a step function, the sigmoid neuron would be a perceptron, since the o/p would be 1 or 0 depending on whether wx+b was +ve or -ve, but sigmoid function has a smooth curves, the smoothness means that a small change Δwj in the weights and Δb in the bias produces small changes o/p in the o/p from the neuron.
    
### Cost Function:

  * The **cost function** or **loss function** gives us the error value our model is giving when trying to classify or predict a cetain data-point's output.
  
  * For network1 we will be using quadratic cost function:
      
     ![cost_function](https://user-images.githubusercontent.com/33928040/68085425-fa3e0a80-fe66-11e9-953b-86ca159e551c.PNG)
    
### Gradient Descent:

  * In machine learning, **gradient descent** is an optimization technique used for computing the model parameters (weights and bias) for algorithms like linear regression, logistic regression, neural networks, etc. In this technique, we repeatedly iterate through the training set and update the model parameters in accordance with the gradient of error with respect to the training set.
  
  * Depending on the number of training examples considered in updating the model parameters, we have 3-types of gradient descents:
    
      1. **Batch Gradient Descent:** Parameters are updated after computing the gradient of error with respect to the entire training set.
      2. **Stochastic Gradient Descent:** Parameters are updated after computing the gradient of error with respect to a single training example.
      3. **Mini-Batch Gradient Descent:** Parameters are updated after computing the gradient of error with respect to a subset of the training set.
      
       **NOTE:** *Mini-Batch Gradient Descent makes a compromise between the speedy convergence and the noise associated with gradient update which makes it a more flexible and robust algorithm.*
  
  * Because of the flexiblity and robustness of the *Mini-Batch* algorithm, we will be using the **mini batch gradient descent** for training our neural network.
                                               
### Backpropagation Algorithm:

  * In machine learning, specifically deep learning, backpropagation is an algorithm widely used in the training of feedforward neural networks for supervised learning.
  
  * Backpropagation is based around four fundamental equations. Together, those equation give us a way of computing both the error and the gradient of the cost function:
  
    1. **An equation for the error in the output layer, δL:** 
         
         ![back1](https://user-images.githubusercontent.com/33928040/68085555-21490c00-fe68-11e9-94c3-fbe723fce164.PNG)   or   ![back1 1](https://user-images.githubusercontent.com/33928040/68085646-1fcc1380-fe69-11e9-9e03-141927a4fd25.PNG)
         
    2. **An equation fo the error δl in terms of the error in the next layer, δl+1:**
         
         ![back2](https://user-images.githubusercontent.com/33928040/68085592-856bd000-fe68-11e9-8b4b-c26b648a763c.PNG)
         
    3. **An equation for the rate of change of the cost with respect to any bias in the network:**
    
         ![back3](https://user-images.githubusercontent.com/33928040/68085612-b9df8c00-fe68-11e9-93aa-5a046bea8765.PNG)

    4. **An equation for the rate of change of the cost with respect to any weight in the network:**
    
         ![back4](https://user-images.githubusercontent.com/33928040/68085627-e4314980-fe68-11e9-9076-ae022985baff.PNG)

  * The backpropagation algorithm provide us with a way of computing the gradient of the cost function by performing the following operations:
    
    ![back_algo](https://user-images.githubusercontent.com/33928040/68085697-bdbfde00-fe69-11e9-9eea-70da2e20bc41.PNG)



***So, these are all the functions and algorithm that we will be using in network1 to build our neural network.***
