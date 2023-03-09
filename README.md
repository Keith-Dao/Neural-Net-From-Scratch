# Neural Net From Scratch

This project is an implementation of a simple neural network with two hidden layers, made specifically to classify digits in the MNIST dataset.

## Goals:

- Create the neural network from scratch, i.e. without the use of deep learning libraries (PyTorch, TensorFlow, etc.)
- Be able to train the model
- Be implemented in Python and C++
- Be able to save and load the model weights

## Architecture Overview

Since an image in the MNIST dataset is a greyscale 28x28 image, the input size of the network will be 784 neurons wide for each pixel in the image. The hidden layers will consist of a linear function fed into a non-linear activation function, in this case a ReLU function, with 250 neurons each. The output layer will consist of a linear function with 10 neurons for each class in the dataset.

Below is a simplified view of the network.

![Neural network architecture](Resources/Neural%20Net.png)
