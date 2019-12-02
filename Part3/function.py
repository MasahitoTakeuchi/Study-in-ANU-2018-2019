import numpy as np


# Activation function
def relu(inputs):
    return np.maximum(inputs, 0)

# Output probability distribution function
def softmax(inputs):
    exp = np.exp(inputs)
    return exp/np.sum(exp, axis = 1, keepdims = True)

# Loss function
def cross_entropy(inputs, y):
    indices = np.argmax(y, axis = 1).astype(int)
    probability = inputs[np.arange(len(inputs)), indices]
    log = np.log(probability)
    loss = -1.0 * np.sum(log) / len(log)
    return loss

# L2 regularization
def L2_regularization(lamda, weight1, weight2):
    weight1_loss = 0.5 * lamda * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * lamda * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss