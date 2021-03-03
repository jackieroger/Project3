import numpy as np

# References used for building the neural network:
# 1) https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
# 2) https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# 3) https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c
# 4) https://pub.towardsai.net/nothing-but-numpy-understanding-creating-neural-networks-with-computational-graphs-from-scratch-6299901091b0


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
        # Initialize layers
        self.input_layer = np.zeros(shape=(1, input_layer_size))
        self.hidden_layer = np.zeros(shape=(1, hidden_layer_size))
        self.output_layer = np.zeros(shape=(1, output_layer_size))
        # Initialize weights as small random numbers
        np.random.seed(self.seed)
        self.weights1 = np.random.rand(input_layer_size, hidden_layer_size)
        np.random.seed(self.seed)
        self.weights2 = np.random.rand(hidden_layer_size, output_layer_size)
        # Initialize biases as 0
        self.biases1 = np.zeros(shape=(1, hidden_layer_size), dtype=float)
        self.biases2 = np.zeros(shape=(1, output_layer_size), dtype=float)

    def _sigmoid(self, j):
        """
        Computes sigmoid activation.

        j : int or float
            A numeric value, arbitrarily denoted j, as in jackie :)
        """
        # Reference: https://www.geeksforgeeks.org/implement-sigmoid-function-using-numpy/
        return 1 / (1 + np.exp(-j))

    def _feedforward(self):
        """
        Feedforward through NeuralNetwork to get predicted values for output.
        """
        self.hidden_layer = self._sigmoid(np.dot(self.input_layer, self.weights1) + self.biases1)
        self.output_layer = self._sigmoid(np.dot(self.hidden_layer, self.weights2) + self.biases2)

    def train(self, input_values, output_values, learning_rate=0.1, num_epochs=10):
        """
        Trains the NeuralNetwork using data with input data and corresponding output data.

        Parameters
        ----------
        input_values : np.array of ints
            Values for input data.
        output_values : np.array of ints
            Values for output data.
        learning_rate : float
            Learning rate for training. Optional. Default is 0.1.
        num_epochs : int
            Number of epochs for training. Optional. Default is 10.

        Returns
        -------
        epoch_losses : np.array of floats
            Average loss per epoch during training.
        """
        # For each epoch
        #for e in range(num_epochs):
            # For each set of input values
            #for i in range(len(input_values)):
                # Get input and output data for this iteration
                #self.input_layer = input_values[i,:]
                #true_output = output_values[i,:]
                # Feedforward
                #self._feedforward()
                #print(self.weights1)
        self.input_layer = input_values[0:2,:]
        self._feedforward()
        print(self.output_layer)


bby_nn = NeuralNetwork(3,2,3)
bby_input = np.array([[0,1,0], [1,1,1], [0,0,0], [1,1,0]])
bby_output = np.array([[0,1,0], [1,1,1], [0,0,0], [1,1,0]])
bby_nn.train(bby_input, bby_output, 0.1, 1)
