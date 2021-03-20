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
    There are two weight matrices: one for going from the input layer to the hidden layer, and another
    for going from the hidden layer to the output layer. Similarly, there are two bias matrices.

    Attributes
    ----------
    seed : int
        Seed for random number generation. Used for reproducibility. Optional. Default is 0.
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

    def train(self, input_values, output_values, batch_size=10, num_epochs=1000, learning_rate=0.1, loss_threshold=0.00000000001):
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
            Number of samples to be used in each batch of training. Optional. Default is 10.
        num_epochs : int
            Number of epochs for training. Optional. Default is 1000.
        learning_rate : float
            Learning rate for training. Optional. Default is 0.00000000001.
        loss_threshold : float or int
            Loss threshold for stopping gradient descent. If loss is below this AND loss is no longer decreasing,
            training stops and it is assumed that reasonable convergence is reached in the learned parameters.
            Optional. Default is 0.001.
        """
        # Save epoch losses
        epoch_losses = []
        # For each epoch
        for e in range(num_epochs):
            # For each batch
            b = 0 #batch index
            l = 0 #batch loss
            while b < input_values.shape[0]:
                # Batch losses
                bl = []
                # Make input layer
                self.input_layer = input_values[b:b+batch_size,:]
                # Forward pass
                self._forward()
                # Compute losses using binary cross entropy
                bce = self._bce(self.output_layer, output_values[b:b+batch_size,:])
                bl.append(bce)
                # Backward pass
                self._backward(self.output_layer, output_values[b:b+batch_size,:], learning_rate)
                b += batch_size
            # Average losses across batches to get average loss for that epoch
            epoch_losses.append(np.mean(bl))
            # Stop criterion for convergence (if loss is not decreasing & it's already small enough)
            if e > 0 and epoch_losses[-1] > epoch_losses[-2] and l < loss_threshold:
                break
        # Save epoch losses as class attribute
        self.epoch_losses = np.array(epoch_losses)

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

def encode(seqs):
    """
    Encodes DNA sequences into a binary sequences using one hot encoding.

    Arguments
    ---------
    seq : np.array of str
        DNA sequences to be encoded.

    Returns
    -------
    bin_seqs : np.array of ints
        Binary encoding of sequences.
    """
    # Make dict for encoding
    enc_map = {"A": np.array([1,0,0,0]),
               "T": np.array([0,1,0,0]),
               "C": np.array([0,0,1,0]),
               "G": np.array([0,0,0,1]),}
    # Do encoding
    bin_seqs = []
    for s in seqs:
        bin_seqs.append(np.array([enc_map[n] for n in s]).flatten())
    return np.array(bin_seqs)

def read_fasta(fasta_file):
    """
    Reads in fasta file and returns sequences.

    Arguments
    ---------
    fasta_file : str
        Name of file.

    Returns
    -------
    seqs : np.array of str
        Sequences from fasta file.
    """
    # Open fasta file
    with open(fasta_file) as ff:
        # Read lines
        lines = ff.readlines()
        # Strip whitespace & make everything uppercase
        lines = [l.strip().upper() for l in lines]
        # Remove any empty strings
        lines = [l for l in lines if l != ""]
        # Check which type of sequence file it is
        if lines[0][0] == ">": # fasta with headers & containing multi-line sequences
            # Make dict of sequences
            seq_dict = {}
            # Make sequence index
            seq_i = -1
            # Parse through lines and add each sequence
            for i in range(len(lines)):
                # If a new header is reached (new seq)
                if lines[i][0] == ">":
                    seq_i += 1
                    seq_dict[seq_i] = ""
                # Concatenate to current seq
                else:
                    seq_dict[seq_i] += lines[i]
            return np.array(list(seq_dict.values()))
        else: # sequence file without headers & containing single-line sequences
            return np.array(lines)

def sample_subseqs(seqs, k, n, exclude=np.array([]), seed=0):
    """
    Randomly subsample subsequences from an array of sequences.

    Arguments
    ---------
    seqs : np.array
        Sequences to be subsampled from.
    k : int
        Length of each subsequence.
    n : int
        Number of subsequences.
    exclude = np.array of str
        Sequences that should be excluded from consideration as an outputted subsequence. Optional. Default is empty string.
    seed : int
        Seed for random number generation. Used for reproducibility. Optional. Default is an empty array.

    Returns
    -------
    subseqs : np.array of str
        Subsequences.
    """
    # Pick n sequences
    np.random.seed(seed)
    seq_indices = np.random.randint(len(seqs), size=(n))
    n_seqs = np.array(seqs)[seq_indices]
    # Make list of subsequences
    subseqs = []
    # Pick kmers from each of the n sequences
    for i in range(n):
        np.random.seed(seed)
        kmer_start = np.random.randint(len(n_seqs[i])-k)
        kmer = n_seqs[i][kmer_start:kmer_start+k]
        while kmer in list(exclude):
            kmer_start = np.random.randint(len(n_seqs[i]) - k)
            kmer = n_seqs[i][kmer_start:kmer_start + k]
        subseqs.append(kmer)
    # Return array of subseqs
    return np.array(subseqs)

