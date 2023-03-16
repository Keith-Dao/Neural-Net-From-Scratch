# Neural Net From Scratch

This project is an implementation of a simple neural network with two hidden layers, made specifically to classify digits in the MNIST dataset.

## 1. Goals:

- Create the neural network from scratch, i.e. without the use of deep learning libraries (PyTorch, TensorFlow, etc.)
- Be able to train the model
- Be implemented in Python and C++
- Be able to save and load the model weights

## 2. Architecture Overview

Since an image in the MNIST dataset is a greyscale 28x28 image, the input size of the network will be 784 neurons wide for each pixel in the image. The hidden layers will consist of a linear function fed into a non-linear activation function, in this case a ReLU function, with 250 neurons each. The output layer will consist of a linear function with 10 neurons for each class in the dataset.

Below is a simplified view of the network.

![Neural network architecture](Resources/Neural%20Net.png)

## 3. Mathematics Overview

### 3.1. Variables

The following are a brief description of all the variables that will appear.

| Variable  | Description                                           |
| --------- | ----------------------------------------------------- |
| $X$       | The input values.                                     |
| $H_i$     | The values of the neurons for the $i$-th hidden layer |
| $O$       | The output values of the network                      |
| $\hat{y}$ | The predicted class                                   |
| $y$       | The actual class                                      |
| $W_x$     | The weights going into the layer $x$                  |
| $B_x$     | The bias for the layer $x$                            |

The following are the dimensions for all the variables.

| Variable  | Dimensions             |
| --------- | ---------------------- |
| $X$       | Minibatch size x $784$ |
| $H_1$     | Minibatch size x $250$ |
| $H_2$     | Minibatch size x $250$ |
| $O$       | Minibatch size x $10$  |
| $\hat{y}$ | Minibatch size x $1$   |
| $y$       | Minibatch size x $1$   |
| $W_{H_1}$ | $784$ x $250$          |
| $W_{H_2}$ | $250$ x $250$          |
| $W_{O}$   | $250$ x $10$           |
| $B_{H_1}$ | $250$                  |
| $B_{H_2}$ | $250$                  |
| $B_{O}$   | $10$                   |
