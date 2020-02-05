# nework2

## Overview:

  * Here in **network2**, we will implement better techniques to improve the performance of our neural network.
  
  * **network2** includes: a better cost function, known as **cross-entropy** cost function; a regularization method known as **L2 Regularization** or **weight decay**.
  
  * **network2** will introduce us with further topics like: **overfitting**, **validation data** and at last the **weight initialization**.
  
  
## Cross-Entropy Cost Function:

### Why to use it?

  * The main question that arises here is: *why are we using cross-entropy cost function in-place for quadratic cost function*.
  
  * The answer to the question lies in the equation itself(the equation of quadratic cost function). We know that our neuron learns by changing weights and biases at a rate determined by the partial derivative of cost function. So let's now try to find the partial derivative of our quadratic cost funtion to see what happens:
  
    ![cost_fun](https://user-images.githubusercontent.com/33928040/68563372-5dd1c480-0473-11ea-87e0-4c414ffb5849.JPG)  
  
    ![cost_derivative](https://user-images.githubusercontent.com/33928040/68563401-7215c180-0473-11ea-977d-d61d8eb8d674.JPG)
    
    **NOTE:** *For these above equations, we have taken x=1 and y=0, that means our neuron will output a zero when the input is one.*
    
  * To understand the behaviour of these expression, let's look more closely to the graph of σ′(z).
  
    ![graph](https://user-images.githubusercontent.com/33928040/68563687-552dbe00-0474-11ea-80b6-6a0b51865b87.JPG)

  * If we look at the graph of σ′(z), we will see, when the neuron output is close to one the value of σ′(z) gets small. So the above equations tells us that the partial derivative of the cost function w.r.t. weight and bias gets very small. Thus, leading to *learning slowdown*.
  
  * Well the learning slowdown can be seen even when we to try to train a perceptron with a large weigth and bias. Below is the graph of *cost v/s epoch* when the values of weight and the bias are taken large:
  
    ![cost_v_epoch](https://user-images.githubusercontent.com/33928040/68564029-78a53880-0475-11ea-88a4-ddce094ecb34.JPG)
 
 * We can see that learning starts out much more slowly. Indeed, for the first 150 or so learning epochs, the weights and biases don't change much at all. Then the learning kicks in, well this is what we call **learning slowdown**.
 
 
### Introducing the cross-entropy cost function

  * Let's take an another neuron with several input variables x1, x2, x3, ...., corresponding weights w1, w2, w3, ..., and a bias b:
    
    
    ![perceptron](https://user-images.githubusercontent.com/33928040/68564443-a76fde80-0476-11ea-9e46-2b4a418857e6.JPG)


  * The output from the neuron is, of course, a=σ(z), where z=∑w\*x + b is the weighted sum of the inputs. We define the cross-entropy cost function for this neuron by
  
    ![cross_entropy](https://user-images.githubusercontent.com/33928040/68564549-eaca4d00-0476-11ea-84c5-cf12e337bcb2.JPG)

    where, *n* is the *total number fo items* of the training data, the sum is over all training inputs *x* and *y* is the corresponding desired output.
    
  * The main benefit of *cross-entropy* function over the *quadratic* function is: **it avoids the problem of learning slowdown.**
  
  * To see it, let's partially derivate the cross-entropy cost function:
  
    ![deri1](https://user-images.githubusercontent.com/33928040/68564766-78a63800-0477-11ea-8233-e09beb112212.JPG)
    
    ![deri2](https://user-images.githubusercontent.com/33928040/68564780-8065dc80-0477-11ea-8265-0eade4ed8b0f.JPG)
    
    ![deri3](https://user-images.githubusercontent.com/33928040/68564788-8956ae00-0477-11ea-811e-cb6a477540f3.JPG)
    
  * The equation tells us that the rate at which the weight learns is controlled by **σ(z)−y**, i.e. *by the error in the output*. The larger the error, the faster the neuron will learn.
  
  * In a similar way, we can compute the partial derivative for the bias:
  
    ![deri_bias](https://user-images.githubusercontent.com/33928040/68564911-f10cf900-0477-11ea-83f8-c37bcd66b784.JPG)

  * Now, it's also easy to generalize the cross entropy to many-neuron multi-layer netowk. In particular, suppose y = y1, y2, ... are desired values at the output neurons, i.e. the neurons in the final layer, while a<sup>L</sup><sub>1</sub>, a<sup>L</sup><sub>2</sub>,… are the actual output values. Then we define the cross-entropy by:
  
    ![cost_neuron](https://user-images.githubusercontent.com/33928040/68565187-bd7e9e80-0478-11ea-9753-8a4578e2228d.JPG)


## Overfitting and Regularization:

### Overfitting

  * A model with a large number of free parameters can describe an amazingly wide range of phenomena. Even if such a model agrees well with the available data, that doesn't make it a good model.
  
  * Well what happens in the model will work well for the existing data, but will fail to generalize to new situations. The true test of a model is it's ability to make predictions in situations it hasn't been exposed to before.
  
  * **Overfitting** or sometimes refers to as **high variance** is cause when our model fits the available data but does not generalize well to predict new data. It is usually caused when we try using lots of features.
  
  * There are quite a number of ways in which we can prevent our model from overfitting.
  
### Use of Validation Data
  
  * This is part of a more general strategy, which is to use the validation data to evaluate different trial choices of hyperparameters such as the number of epochs to train for, the learning rate, the best n/w architecture and so on. We use such evaluations to find and set good values for the hyperparameters.
  
  * To understand why, consider that when setting hyper-parameter we're likely to try many different choices for the hyper-parameters. If we set the hyper-parameter based on evaluations of the test data it's possible we'll end up overfitting our hyper-parameters to the test data. That is, we may end up finding hyper-parameters which fit particular pecularities of the test-data, but where the performance of the network won't generalize to the other data sets.
  
### Regularization

  * **Regularization** is a technique used for tuning the function by adding an additional penalty term in the error function. The additional term controls the excessively fluctuating function such that the coefficients(weights and biases) don't take extreme values.
  
  * **L2 Regularization** or **weight-decay**, here is the updated cost-function which is regularized(L2):
    
    ![regucost](https://user-images.githubusercontent.com/33928040/68566065-5ca49580-047b-11ea-9faa-9bb3544490f7.JPG)
    
    The first term is just the usual expression for the cross-entropy. But we've added a second term, namely the sum of the squares of all the weights in the network. This is scaled by a factor λ/2n, where λ>0 is known as the regularization parameter, and n is, as usual, the size of our training set. 
    
  * It is known as **weight decay** because it makes the weights smaller. We can see that by applying our regularized cost function to our learning rule, for bias it remains the same, but for weight the equation changes to:
  
    ![weight_eq](https://user-images.githubusercontent.com/33928040/68566224-b86f1e80-047b-11ea-8067-2037ca3fe3e3.JPG)

  * Here, we are rescaling the weight w by a factor of 1−(ηλ/n), and this rescaling is sometime referred as **weight decay**.
  
## Neural Network's Ouput:

  * Our neural network has two hidden layer having 30 neurons each.(you can change the value, you only have to edit the code of run_script.py file)
  
  * We have 784 i/p neurons and 10 output neurons.
  
  * The final accuracy we are getting is around 95%.
  
  ![acc_`](https://user-images.githubusercontent.com/33928040/73866201-75e69380-486a-11ea-8169-38a1babed89c.PNG)
  ![acc_2](https://user-images.githubusercontent.com/33928040/73866234-8434af80-486a-11ea-8084-f25f91150874.PNG)
  ![acc_3](https://user-images.githubusercontent.com/33928040/73866262-8dbe1780-486a-11ea-8d26-5d065440a77a.PNG)

