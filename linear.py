import numpy as np

from layer import Layer


class LinearLayer(Layer):

    def __init__(self, input_size, output_size, l1_lambda=0, l2_lambda=0):
        super().__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.input_size = input_size
        self.output_size = output_size
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # L1 regularization term
        if self.l1_lambda > 0:
            weights_error += self.l1_lambda * np.sign(self.weights)
        # L2 regularization term
        if self.l2_lambda > 0:
            weights_error += self.l2_lambda * self.weights
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    def get_regularization_loss(self):
        l1_loss = 0
        if self.l1_lambda > 0:
            l1_loss = self.l1_lambda * np.sum(np.abs(self.weights))
        l2_loss = 0
        if self.l2_lambda > 0:
            l2_loss = 0.5 * self.l2_lambda * np.sum(self.weights ** 2)
        return l1_loss + l2_loss
