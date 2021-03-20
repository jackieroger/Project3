from scripts import NN
import numpy as np

# Test that network architecture is correctly set up and that all matrices are correct
# dimensions after training (as a sanity check for all the matrix multiplications)
def test_nn_setup():
    # Make nn
    bby_nn = NN.NeuralNetwork(3, 4, 3)
    bby_input = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]])
    bby_output = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]])
    bby_nn.train(bby_input, bby_output, 4)
    # Check that the layers, weights, & biases have the correct dimensions
    assert bby_nn.input_layer.shape == (4, 3)
    assert bby_nn.hidden_layer.shape == (4, 4)
    assert bby_nn.output_layer.shape == (4, 3)
    assert bby_nn.weights1.shape == (3, 4)
    assert bby_nn.weights2.shape == (4, 3)
    assert bby_nn.biases1.shape == (1, 4)
    assert bby_nn.biases2.shape == (1, 3)

# Check random initialization of weights & biases
def test_random_WB_init():
    # Make nn
    bby_nn = NN.NeuralNetwork(7, 3, 5)
    # Check the random initialization by checking that all the values are unique
    # (a random seed is set during the init of the nn, so the matrices generated during
    # this test are consistent with each test run and will never randomly include 2
    # identical values in an matrix)
    assert len(np.unique(bby_nn.weights1)) == 21
    assert len(np.unique(bby_nn.weights2)) == 15
    assert len(np.unique(bby_nn.biases1)) == 3
    assert len(np.unique(bby_nn.biases2)) == 5

# Test one hot encoding of DNA sequence
def test_encoding():
    # Make seq
    bby_seq = np.array(["ATCG"])
    # Do encoding
    bby_bin_enc = NN.encode(bby_seq)
    # Check encoding
    assert np.array_equal(bby_bin_enc, np.array([[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]]))

# Test fasta reading
def test_fasta_reading():
    # Positive seqs
    pos_seqs = NN.read_fasta("data/rap1-lieb-positives.txt")
    # Negative seqs
    neg_seqs = NN.read_fasta("data/yeast-upstream-1k-negative.fa")
    # Sanity checks
    assert len(pos_seqs) == 137
    assert len(neg_seqs) == 3164
    assert len(pos_seqs[0]) == 17
    assert len(neg_seqs[0]) == 1000

# Test sequence subsampling
def test_subseq():
    # Read in seqs
    seqs = NN.read_fasta("data/yeast-upstream-1k-negative.fa")
    # Get subseqs
    subseqs = NN.sample_subseqs(seqs, 5, 3)
    # Get subseqs excluding the sequence "CACTA"
    excl = np.array(["CACTA"])
    subseqs_ex = NN.sample_subseqs(seqs, 5, 3, exclude=excl)
    # Sanity checks
    assert len(subseqs) == 3
    assert len(subseqs[0]) == 5
    assert len(subseqs_ex) == 3
    assert len(subseqs_ex[0]) == 5
    # Explicitly check that sequence exclusion works
    # (this is consistent because of seed set in function)
    assert subseqs[0] == "CACTA"
    assert subseqs_ex[0] != "CACTA"

# Test network by checking that weights changed and losses decreased
def test_network():
    # Make nn
    bby_nn = NN.NeuralNetwork(3, 2, 3)
    bby_input = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]])
    bby_output = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]])
    # Save a weight matrix
    bby_weights = bby_nn.weights1
    bby_nn.train(bby_input, bby_output, batch_size=4, num_epochs=5)
    # Check that losses decreased
    assert bby_nn.epoch_losses[4] < bby_nn.epoch_losses[0]
    # Check that weights changed
    assert not np.array_equal(bby_nn.weights1, bby_weights)