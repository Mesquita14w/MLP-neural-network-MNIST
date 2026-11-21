import pandas as pd
import numpy as np
import os

DATA_DIR = "./data"

"""
This is the main file of the project, it contains all the MLP class
and some math functions that are used in the MLP class.

I use run code with all the steps in my local IDE (is i mentioned 
earlier, i don't like to use Jupyter Notebooks.), but you can use
whereaver you want.

All the metrics and results of the model are in the repository.

This code just contains the model architecture.
"""

# Load the data:

train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')

raw_train = pd.read_csv(train_path).to_numpy()
y_all = raw_train[:, 0].astype(np.int64)
X_all = raw_train[:, 1:].T / 255.0

if os.path.exists(test_path):
    raw_test = pd.read_csv(test_path).to_numpy()
    X_kaggle_test = raw_test.T / 255.0  # Shape (784, N_test)