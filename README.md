# Project 3 - Neural Networks
## Due 03/19/2021

![BuildStatus](https://github.com/jackieroger/Project3/workflows/HW3/badge.svg?event=push)

### Layout of repo

The code for my implementation of a neural network for this assignment
is located in scripts/NN.py. The code for my unit tests is located in
test/test_NN.py. The sphinx documentation for my API is located in
docs/_build/html/index.html. The output of my model from the final test set (part 5)
is in the file part5-results.tsv. Everything else (written responses,
autoencoder, analysis of TF binding site sequence data, cross validation,
optimization, etc) is in the jupyter notebook titled Jackie_Roger_BMI203_HW3.ipynb.

### Testing

To run all unit tests, run the following command from the root directory:
```
python -m pytest test/*
```

### Usage example
(This is for illustrative purposes only. Please do not build a model using a dataset this small.)
```
from scripts import NN
import numpy as np
from sklearn.metrics import roc_auc_score

# Create a 5x3x1 neural network
example_nn = NN.NeuralNetwork(5,3,1)

# Train model
training_data_input = np.array([[1,2,3,4,5], [3,3,3,3,3]])
training_data_output = np.array([[0], [1]])
example_nn.train(training_data_input, training_data_output)

# Apply model to validation data & evaluate performance
validation_data_input = np.array([[2,5,2,5,2], [1,1,1,4,4]])
validation_data_output = np.array([[1], [0]])
validation_data_predictions = example_nn.predict(validation_data_input)
validation_data_auc = roc_auc_score(validation_data_output, validation_data_predictions)
```
