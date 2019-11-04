"""
mnist_loader
------------

A library to load the MNIST image data. The mnist_loader library
includes three functions: load_data, load_data_wrapper and vectrorized_result.

load_data function is used for opening the gz archive file (our MNIST dataset),
loading and returning the training, validation and test dataset.

The load_data_wrapper function actually returns 
the training, validation and test data in a required format
which is accepted by our neural network.

vectorized_result function is used for returing a
10-dimensional vector which will be output for particular input.

In practice, the load_data_wrapper function is the function
that is usually called by our neural network.
"""

## Libraries
"""
We will be using two basic Python in-built libraries
for our mnist_loader library: _pickle and gzip.

_pickle: The pickle module implements an algorithm for
turning an arbitrary Python object into a series of bytes. The cPickle module
implements the same algorithm but in C instead of Python.
On python3.x cPickle has changed from cPickle to _pickle.

gzip: This module provides a simple interface to compress and
decompress files just like the GNU program gzip and gunzip would.

Apart from these two basic libraries we will be using numpy
library as well for all the numerical computations that are required
to be performed by the functions in mnist_loader library.
"""
import _pickle as cPickle    ## cPickle implementation in python3.x
import gzip

import numpy as np

def load_data():
    """ Return the MNIST data as a tuple containing
    the training data, the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.
    This is a numpy ndarray(N-dimensional array) with 50,000 enties.
    Each entry is, in turn, a numpy ndarray with 784 values,
    representing the 28x28 = 784 pixels in a single MNIST image.

    The second entry in the ``training_data`` is a numpy ndarray
    containing 50,000 entries. Those entries are just
    the digit values (0...9) for the corresponding images
    contained in the first entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar,
    except each contains only 10,000 images.

    This is a nice data format, but for our neural network
    to use the values we need to modify the format of
    the ``training_data`` a little. That is why
    we have another function(see below) named as ``load_data_wrapper()``.
    """

    mnist_file = gzip.open('../data/mnist.pkl.gz', 'rb')
    ## opening the gz archive file by using gzip's open function

    training_data, validation_data, test_data = cPickle.load(mnist_file, encoding='latin1')
    ## loading the training, validation and test data by using cPickle's load function
    ## passing encoding parameter as ``latin1``

    mnist_file.close()
    ## closing the mnist_file

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    Return a tuple containing ``(training_data, validation_data, test_data)``.
    We have made this function to make the format of
    the training dataset a more convenient one to be used
    in our implementation of neural network.

    In particular, ``training_data`` is a list containing
    50,000 2-tuple ``(x,y)``. ``x`` is a 784-dimensional numpy.ndarray
    containing the input image. ``y`` is a 10-dimensional numpy.ndarray
    representing the unit vector coreesponding to the correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing
    10,000 2-tuples ``(x,y)``. In each case, ``x`` is a 784-dimensional
    numpy.ndarray containing the input image, and
    ``y`` is the corresponding i.e., the digit values(integers) corresponding to ``x``.

    Well these above formats are the formats that are required by
    our neural network, so our function ``load_data_wrapper``
    will return each of the dataset in these required format,
    and these formats turn out to be the most convenient for use in our neural network.

    One thing to note here, the output of the ``training_data``
    is a 10 dimensional vector but in case of ``validation_data/test_data``
    our output is just the digit value corresponding to the input ``x``.
    """

    train_data, valid_data, tst_data = load_data()
    ## calling the function load_data()
    ## will return a tuple with three values for train, validation and test data
    ## storing the tuple values in separate three variables

    ## training_data:
    training_inputs = [np.reshape(x, (784,1)) for x in train_data[0]]
    ## reshaping the training inputs to 784x1 vector
    ## the required format for our neural network's input layer
    ## ---
    training_results = [vectorized_result(y) for y in train_data[1]]
    ## calling vectorized_result() function(see below)
    ## will convert the digit value in 10-dimensional vector
    ## the required format for our neural network's output layer
    ## ---
    training_data = zip(training_inputs, training_results)
    ## zipping together the training_inputs and training_results

    ## validation_data:
    validation_inputs = [np.reshape(x, (784,1)) for x in valid_data[0]]
    ## reshaping the validation inputs to 784x1 vector
    ## ---
    validation_data = zip(validation_inputs, valid_data[1])
    ## zipping together the validation_inputs and it's corresponding outputs

    ## test_data:
    test_inputs = [np.reshape(x, (784,1)) for x in tst_data[0]]
    ## reshaping the test inputs to 784x1 vector
    ## ---
    test_data = zip(test_inputs, tst_data[0])
    ## zipping together the test_inputs and it's corresponding outputs

    return (training_data, validation_data, test_data)

def vectorized_result(y):
    """
    Return a 10-dimensional unit vector with a 1.0 value at
    the yth position and zeroes elsewhere.

    This is used to convert a digit (0...9) into a corresponding
    desired output from the neural network.

    e.g. if the digit is 6 the 10-dimensional unit vector will be:
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]T
    T is the transpose of the given array making it a 10x1 dimensional unit vector
    """

    e = np.zeros((10,1))
    ## making an array of length 10 having zero in all the indices

    e[y] = 1.0
    ## at index y assigning the value 1.0

    return e
