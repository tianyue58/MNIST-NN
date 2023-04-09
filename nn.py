import random
import numpy as np

from linear import LinearLayer


class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss):
        self.loss = loss

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, initial_learning_rate=0.01, verbose=1):
        train_samples = len(X_train)
        val_samples = len(X_val)
        train_errors = []
        val_errors = []
        reg_losses = []
        total_losses = []
        train_accuracies = []
        val_accuracies = []
        learning_rate = initial_learning_rate
        decay_rate = learning_rate / epochs
        for i in range(epochs):
            train_error = 0
            val_error = 0
            reg_loss = 0
            train_correct = 0
            val_correct = 0
            # Train on mini-batches
            for j in range(0, train_samples):
                # Stochastic Gradient Descent (SGD)
                # Select a random sample from the training data
                index = random.randint(0, train_samples - 1)
                # Forward pass
                output = X_train[index]
                for layer in self.layers:
                    output = layer.forward(output)
                train_error += self.loss(y_train[index], output)
                train_correct += int(np.argmax(y_train[index]) == np.argmax(output))
                # Backward pass
                error = self.loss(y_train[index], output, derivative=True)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
                    reg_loss += layer.get_regularization_loss()
            # Evaluate on validation dataset
            for j in range(0, val_samples):
                # Forward pass
                output = X_val[j]
                for layer in self.layers:
                    output = layer.forward(output)
                val_error += self.loss(y_val[j], output)
                val_correct += int(np.argmax(y_val[j]) == np.argmax(output))
            # Compute average error for this epoch
            train_error /= train_samples
            val_error /= val_samples
            reg_loss /= train_samples
            total_loss = train_error + reg_loss
            train_accuracy = train_correct / train_samples
            val_accuracy = val_correct / val_samples
            if verbose != 0:
                print("epoch %d/%d: train_loss = %f, val_loss = %f, reg_loss = %f, "
                      "total_loss = %f, train_accuracy = %f, val_accuracy = %f"
                      % (i + 1, epochs, train_error, val_error, reg_loss, total_loss, train_accuracy, val_accuracy))
            train_errors.append(train_error)
            val_errors.append(val_error)
            reg_losses.append(reg_loss)
            total_losses.append(total_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            learning_rate = learning_rate * (1 - decay_rate)
        return train_errors, val_errors, reg_losses, total_losses, train_accuracies, val_accuracies

    def predict(self, input_data):
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result

    def summary(self):
        total_params = 0
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                layer_params = layer.weights.size + layer.bias.size
                total_params += layer_params
                print(layer.__class__.__name__, "(", layer.input_size, "->", layer.output_size, ")")
                print("Weights:", layer.weights.shape)
                print("Biases:", layer.bias.shape)
                print("Total parameters:", layer_params)
            else:
                print(layer.__class__.__name__)
            print()
        print("Total parameters in model:", total_params)

    def details(self):
        print("Neural Network Details")
        print("=" * 30)
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            print(f"Layer {i}: {layer_name}")
            if isinstance(layer, LinearLayer):
                print("  Weights:")
                print(layer.weights)
                print("  Biases:")
                print(layer.bias)
            else:
                print("  Details not available for this layer type")
            print()
        print("=" * 30)
