import numpy as np

class activation:
    #features
    def __init__(self):
        pass
    def add(self, x, y):
        return x + y
    #softmax with standardizing option
    def softmax(self, array, std):
        max = np.max(array)
        if std == True:
            return np.exp(array - max) / np.sum(np.exp(array - max))
        return np.exp(array) / np.sum(np.exp(array))
    #tangent hyperbolic 
    def tanh(self, array):
        return (np.exp(array) - np.exp(-1 * array)) / (np.exp(array) + np.exp(-1 * array))
    #logistic sigmoid
    def sigmoid(self, array):
        return 1 / (1 + np.exp(-1 * array))