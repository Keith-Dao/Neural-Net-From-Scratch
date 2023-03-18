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
| $P$       | The probability of predicting a class                 |
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
| $P$       | Minibatch size x $10$  |
| $\hat{y}$ | Minibatch size x $1$   |
| $y$       | Minibatch size x $1$   |
| $W_{H_1}$ | $784$ x $250$          |
| $W_{H_2}$ | $250$ x $250$          |
| $W_{O}$   | $250$ x $10$           |
| $B_{H_1}$ | $250$                  |
| $B_{H_2}$ | $250$                  |
| $B_{O}$   | $10$                   |

### 3.2. Forward Pass

The following are functions that will be used during the forward pass.

| Function        | Equation                        |
| --------------- | ------------------------------- |
| Linear function | $$Y = XW + B $$                 |
| ReLU            | $$\sigma(X)_i = \max (X_i, 0)$$ |

The following are the functions in the order preformed by the network.

$$
    H_1 = \sigma (W_{H_1}X + B_{H_1}) \\
    H_2 = \sigma (W_{H_2}H_1 + B_{H_2}) \\
    O = W_O H_2 + B_O
$$

In order to obtain the predicted class, the softmax function must first be applied to the output logits. Then, apply argmax to determine the neuron with the highest probability.

| Function | Equation                                                |
| -------- | ------------------------------------------------------- |
| Softmax  | $$\sigma(X)_i = \frac{e^{X_i}}{\sum _{j=1}^K e^{X_j}}$$ |
| Argmax   | $$ \hat{y} = \argmax\_{i} X_i $$                        |

Thus, the following is applied to obtained the predictions.

$$
    \sigma(O)_i = \frac{e^{O_i}}{\sum_{j=1}^K e^{O_j}} \\
    P = \sigma(O) \\
    \hat{y} = \argmax_i \sigma(O)_i
$$
