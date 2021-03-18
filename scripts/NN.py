import numpy as np

# NEURAL NETWORK CLASS

# References used for building the neural network:
# 1) https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
# 2) https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# 3) https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c
# 4) https://pub.towardsai.net/nothing-but-numpy-understanding-creating-neural-networks-with-computational-graphs-from-scratch-6299901091b0
# 5) https://sudeepraja.github.io/Neural/
# 6) https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795

class NeuralNetwork:
    """
    Represents a neural network containing an input layer, a hidden layer, and an output layer.

    Attributes
    ----------
    seed : int
        Seed for random number generation. Used for reproducibility.
    input_layer : np.array of ints
        The input layer of the network, where each element represents a node in the layer.
    hidden_layer : np.array of ints
        The hidden layer of the network, where each element represents a node in the layer.
    output_layer : np.array of ints
        The output layer of the network, where each element represents a node in the layer.
    weights1 : np.array of floats
        Weights between the input layer and the hidden layer.
    weights2 : np.array of floats
        Weights between the hidden layer and the output layer.
    biases1 : np.array of floats
        Biases between the input layer and the hidden layer.
    biases2 : np.array of floats
        Biases between the hidden layer and the output layer.
    epoch_losses : np.array of floats
        Average loss per epoch during training.
    """
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, seed=0):
        """
        Initializes a NeuralNetwork object using information about the size of the layers.

        Parameters
        ----------
        input_layer_size : int
            The size of the input layer.
        hidden_layer_size : int
            The size of the hidden layer.
        output_layer_size : int
            The size of the output layer.
        seed : int
            Seed for random number generation. Optional. Default is 0.
        """
        # Save seed
        self.seed = seed
        # Initialize weights & biases as small random numbers
        np.random.seed(self.seed)
        self.weights1 = np.random.randn(input_layer_size, hidden_layer_size) * np.sqrt(2.0/8)
        np.random.seed(self.seed)
        self.weights2 = np.random.randn(hidden_layer_size, output_layer_size) * np.sqrt(2.0/8)
        np.random.seed(self.seed)
        self.biases1 = np.random.randn(1, hidden_layer_size) * np.sqrt(2.0/8)
        np.random.seed(self.seed)
        self.biases2 = np.random.randn(1, output_layer_size) * np.sqrt(2.0/8)

    def _sigmoid(self, j):
        """
        Computes sigmoid activation.

        j : int, float, or np.array
            A numeric value, arbitrarily denoted j, as in jackie :)
        """
        return 1 / (1 + np.exp(-j))

    def _forward(self):
        """
        Forward pass through NeuralNetwork to get predicted values for output.
        """
        # Update hidden layer & output layer
        self.hidden_layer = self._sigmoid(np.dot(self.input_layer, self.weights1) + self.biases1)
        self.output_layer = self._sigmoid(np.dot(self.hidden_layer, self.weights2) + self.biases2)

    def _bce(self, preds, trues):
        """
        Compute loss using binary cross entropy.

        Parameters
        ----------
        preds : np.array of ints
            Predicted values.
        trues : np.array of ints
            True values.

        Returns
        -------
        bce : float
            Binary cross entropy loss.
        """
        # Number of values
        m = preds.shape[1]
        # Set epsilon & update preds to avoid any log(0) issues or 0 divide issues later on
        # Credit: the clipping was parker's idea
        eps = np.finfo(float).eps
        preds = np.clip(preds, eps, 1 - eps)
        # Calculate loss
        bce = (1 / m) * np.sum((-trues * np.log(preds) - (1 - trues) * np.log(1 - preds)))
        return np.squeeze(bce)

    def _bce_grad(self, preds, trues):
        """
        Computes the gradient of the binary cross entropy loss function.

        Parameters
        ----------
        preds : np.array of ints
            Predicted values.
        trues : np.array of ints
            True values.

        Returns
        -------
        bce_grad : float
            Gradient of the binary cross entropy loss function.
        """
        # Number of values
        m = preds.shape[1]
        # Set epsilon & update preds to avoid any log(0) issues or 0 divide issues later on
        # Credit: the clipping was parker's idea
        eps = np.finfo(float).eps
        preds = np.clip(preds, eps, 1 - eps)
        # Compute gradient of loss
        bce_grad = (1 / m) * (-(trues / preds) + ((1 - trues) / (1 - preds)))
        return bce_grad

    def _backward(self, preds, trues, learning_rate):
        """
        Backward pass through NeuralNetwork to get values for deltas.

        Parameters
        ----------
        preds : np.array of ints
            Predicted values.
        trues : np.array of ints
            True values.
        learning_rate : float
            Learning rate for training.
        """
        # Note: doing the matrix implementation as described in: https://sudeepraja.github.io/Neural/
        # (the notation in some of these variables is intended to match the notation in this link)
        # CALCULATE DELTAS
        # delta 2
        # Derivative of binary cross entropy loss function comparing output layer to true values
        bce_grad = self._bce_grad(preds, trues)
        # fprime2 = derivative of sigmoid(dot(W2, hidden layer)+B2)
        #         = sigmoid(dot(W2, hidden layer)+B2) * (1 - sigmoid(dot(W2, hidden layer)+B2))
        #         = output layer * (1 - output layer)
        fprime2 = self.output_layer * (1 - self.output_layer)
        # Hadamard product of bce_der and fprime2
        delta2 = bce_grad * fprime2
        # delta 1 (similar logic to delta 1, except use w2_d2 instead of bce_der
        w2_d2 = np.dot(self.weights2, delta2.T).T
        fprime1 = self.hidden_layer * (1 - self.hidden_layer)
        delta1 = w2_d2 * fprime1
        # UPDATE WEIGHTS
        # weights 2
        dEdW2 = np.dot(delta2.T, self.hidden_layer).T
        self.weights2 = self.weights2 - (learning_rate * dEdW2)
        # weights 1
        dEdW1 = np.dot(delta1.T, self.input_layer).T
        self.weights1 = self.weights1 - (learning_rate * dEdW1)
        # UPDATE BIASES
        self.biases2 = self.biases2 - (learning_rate * np.sum(delta2, axis=0))
        self.biases1 = self.biases1 - (learning_rate * np.sum(delta1, axis=0))

    def train(self, input_values, output_values, batch_size, num_epochs=10, learning_rate=0.1):
        """
        Trains the NeuralNetwork using data with input data and corresponding output data using batched gradient
        descent.

        Parameters
        ----------
        input_values : np.array of ints
            Values for input data.
        output_values : np.array of ints
            Values for output data.
        batch_size : int
            Number of samples to be used in each batch of training.
        num_epochs : int
            Number of epochs for training. Optional. Default is 10.
        learning_rate : float
            Learning rate for training. Optional. Default is 0.1.
        """
        # Save epoch losses
        epoch_losses = []
        # For each epoch
        for e in range(num_epochs):
            # For each batch
            b = 0 #batch index
            l = 0 #batch loss
            while b < input_values.shape[0]:
                # Make input layer
                self.input_layer = input_values[b:b+batch_size,:]
                # Forward pass
                self._forward()
                # Compute losses using binary cross entropy
                bce = self._bce(self.output_layer, output_values[b:b+batch_size,:])
                l += bce
                # Backward pass
                self._backward(self.output_layer, output_values[b:b+batch_size,:], learning_rate)
                b += batch_size
            epoch_losses.append(l)
        # Save epoch losses as class attribute
        self.epoch_losses = epoch_losses

    def predict(self, input_values):
        """
        Given a set of input values, predicts output values using a trained NeuralNetwork.

        Parameters
        ----------
        input_values : np.array of ints
            Values for input data.

        Returns
        -------
        predictions : np.array of ints
            Predictions based on input data (this is equivalent to the output layer).
        """
        # Save input layer
        self.input_layer = input_values
        # Do forward pass
        self._forward()
        # Return predictions (output layer)
        return self.output_layer

# UTILITY FUNCTIONS

def encode(seq):
    """
    Encodes DNA sequence into a binary sequence using one hot encoding.

    Arguments
    ---------
    """
    pass