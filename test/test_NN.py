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

