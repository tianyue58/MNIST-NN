from layer import Layer


class ActivationLayer(Layer):

    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return self.activation(self.input, derivative=True) * output_error

    def get_regularization_loss(self):
        return 0
