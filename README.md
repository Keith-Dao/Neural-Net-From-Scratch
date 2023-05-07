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

| Variable  | Description                                                                                   |
| --------- | --------------------------------------------------------------------------------------------- |
| $X$       | The input values.                                                                             |
| $Z_i$     | The values of the neurons for the $i$-th hidden layer before applying the activation function |
| $H_i$     | The values of the neurons for the $i$-th hidden layer                                         |
| $O$       | The output values of the network                                                              |
| $P$       | The probability of predicting a class                                                         |
| $T$       | The one-hot encoded truth label                                                               |
| $\hat{y}$ | The predicted class                                                                           |
| $y$       | The actual class                                                                              |
| $W_x$     | The weights going into the layer $x$                                                          |
| $B_x$     | The bias for the layer $x$                                                                    |
| $n$       | The number of classes, in this case it will always be 10                                      |

The following are the dimensions for all the variables.

| Variable  | Dimensions             |
| --------- | ---------------------- |
| $X$       | Minibatch size x $784$ |
| $Z_1$     | Minibatch size x $250$ |
| $Z_2$     | Minibatch size x $250$ |
| $H_1$     | Minibatch size x $250$ |
| $H_2$     | Minibatch size x $250$ |
| $O$       | Minibatch size x $10$  |
| $P$       | Minibatch size x $10$  |
| $T$       | Minibatch size x $10$  |
| $\hat{y}$ | Minibatch size x $1$   |
| $y$       | Minibatch size x $1$   |
| $W$       | No. out x No. in       |
| $W_{H_1}$ | $250$ x $784$          |
| $W_{H_2}$ | $250$ x $250$          |
| $W_{O}$   | $10$ x $250$           |
| $B$       | No. out                |
| $B_{H_1}$ | $250$                  |
| $B_{H_2}$ | $250$                  |
| $B_{O}$   | $10$                   |

### 3.2. Forward Pass

The following are functions that will be used during the forward pass.

| Function        | Equation                        |
| --------------- | ------------------------------- |
| Linear function | $$Y = XW^T + B $$               |
| ReLU            | $$\sigma(X)_i = \max (X_i, 0)$$ |

The following are the functions in the order preformed by the network.

$$
    Z_1 = X W_{H_1}^T + B_{H_1}
$$

$$
    H_1 = \text{ReLU}(Z_1)
$$

$$
    Z_2 = H_1 W_{H_2}^T + B_{H_2}
$$

$$
    H_2 = \text{ReLU}(Z_2)
$$

$$
    O = H_2 W_O^T + B_O
$$

In order to obtain the predicted class, the softmax function must first be applied to the output logits. Then, apply argmax to determine the neuron with the highest probability.

| Function | Equation                                                                |
| -------- | ----------------------------------------------------------------------- |
| Softmax  | $$\sigma(X) = \frac{e^{X}}{\underset{i=1}{\overset{n}{\sum}} e^{X_i}}$$ |
| Argmax   | $$\hat{y} = \underset{i}{\text{arg max }} X$$                           |

Thus, the following is applied to obtained the predictions.

$$
    P = \frac{e^{O}}{\underset{i=1}{\overset{n}{\sum}} e^{O_i}}
$$

$$
    \hat{y} = \underset{i}{\text{arg max }} \sigma(O)
$$

### 3.3. Backward Pass

The following are functions that will be used during the backward pass.

| Function           | Equation                                                               |
| ------------------ | ---------------------------------------------------------------------- |
| Softmax            | $$\sigma(X) = \frac{e^{X}}{\underset{i=1}{\overset{n}{\sum}}e^{X_i}}$$ |
| Cross-entropy loss | $$L_{CE} = -\sum_{i=1}^n T_i \log (P_i)$$                              |

We will use backpropagation to update the weights, given as the following.

$$
    w = w - \alpha \cdot \frac{\partial L }{\partial w}
$$

$\alpha$ in the equation above is the learning rate. A learning rate of $10^{-4}$ will be used as the default.

Additionally, we need to the gradients of the loss function with respect to the parameter that we are attempting to update. We can obtain the gradients via differentiating and applying the chain rule.

| Function Derivative                                     | Equation  |
| ------------------------------------------------------- | --------- |
| $$\frac{\partial L_{CE}}{\partial O}$$                  | $$P - T$$ |
| $$\frac{\partial \text{ Linear function}}{\partial W}$$ | $$X$$     |
| $$\frac{\partial \text{ Linear function}}{\partial B}$$ | $$I$$     |
| $$\frac{\partial \text{ Linear function}}{\partial X}$$ | $$W$$     |
| $$\frac{\partial \text{ ReLU}}{\partial X}$$            | $$X > 0$$ |

| Parameter | Derivative Chain                                                                                     | Loss Derivative w.r.t Parameter                                    |
| --------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| $O$       | $$\frac{\partial L_{CE}}{\partial O} $$                                                              | $$P - T$$                                                          |
| $W_O$     | $$\left(\frac{\partial L_{CE}}{\partial O}\right)^T \cdot \frac{\partial O}{\partial W_O} $$         | $$\left(\frac{\partial L_{CE}}{\partial O}\right)^T \cdot H_2$$    |
| $B_O$     | $$\frac{\partial L_{CE}}{\partial O} \cdot \frac{\partial O}{\partial B_O}$$                         | $$\sum \frac{\partial L_{CE}}{\partial O}$$                        |
| $H_2$     | $$\frac{\partial L_{CE}}{\partial O} \cdot \frac{\partial O}{\partial H_2}$$                         | $$\frac{\partial L_{CE}}{\partial O} \cdot W_O$$                   |
| $Z_2$     | $$\frac{\partial L_{CE}}{\partial H_2} \cdot \frac{\partial H_2}{\partial Z_2}$$                     | $$\frac{\partial L_{CE}}{\partial H_2} \cdot \{Z_2 > 0\} $$        |
| $W_{H_2}$ | $$\left(\frac{\partial L_{CE}}{\partial Z_2}\right)^T \cdot \frac{\partial Z_2}{\partial W_{H_2}}$$  | $$\left(\frac{\partial L_{CE}}{\partial Z_2} \right)^T \cdot H_1$$ |
| $B_{H_2}$ | $$\frac{\partial L_{CE}}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial B_{H_2}}$$                 | $$\sum \frac{\partial L_{CE}}{\partial Z_2}$$                      |
| $H_1$     | $$\frac{\partial L_{CE}}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial H_1}$$                     | $$\frac{\partial L_{CE}}{\partial Z_2} \cdot W_{H_2}$$             |
| $Z_1$     | $$\frac{\partial L_{CE}}{\partial H_1} \cdot \frac{\partial H_1}{\partial Z_1}$$                     | $$\frac{\partial L_{CE}}{\partial H_1} \cdot \{Z_1 > 0\}$$         |
| $W_{H_1}$ | $$\left(\frac{\partial L_{CE}}{\partial Z_1} \right)^T \cdot \frac{\partial Z_1}{\partial W_{H_1}}$$ | $$\frac{\partial L_{CE}}{\partial Z_1} \cdot X$$                   |
| $B_{H_1}$ | $$\frac{\partial L_{CE}}{\partial Z_1} \cdot \frac{\partial Z_1}{\partial B_{H_1}}$$                 | $$\sum \frac{\partial L_{CE}}{\partial Z_1}$$                      |

For the explanation for the derivative $\displaystyle \frac{\partial L_{CE}}{\partial O}$, view [here](https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1).

# 4. Remarks

The following are some issues that arose during training and evaluation and possible solutions to combat them. These issues are beyond the scope of this project and are merely suggestions to improve the model if it were to be completed again in a deep learning framework.

## 4.1. Optimizer

Since the learning rate remains static throughout training, a fairly small learning rate would need to be used to prevent overshooting. This provides better convergence to a minima, at the cost of requiring more epochs and potentially becoming stuck in a non-global minima. An optimizer like Adam would alleviate these issues by automatically adjusting learning rates per parameter, which reduces the total number of epochs required for convergence without the trade-off of overshooting.

## 4.2. Model performance

Although the model's performs quite well, it can be further improved with the use of 2D convolutional layer to reduce model size via layer weights while better capturing spatial relations.

Although overfitting was not much of an issue for the trained model, there are a couple solutions that can be applied if it does occur. These solutions include dropout layers, data augmentation and weight decay, which all attempt to help the model generalise.

The dropout layers can be slotted in between layers during training and randomly set some of the next layer's inputs to zero.

Data augmentation, such as slight rotations, translations, scale and addition of noise could be randomly added to the training dataset to help broader the range of data that the model is trained on.

Weight decay is a hyperparameter that can be introduced during the backwards pass of the model to penalise parameters with large magnitudes, forcing the model to generalise.
