import random as rn
# Seed for python random numbers.
rn.seed(1254)

import numpy as np
# Seed for numpy-generated random numbers.
np.random.seed(12)

import os
# Seed for certain hash-based algorithms.
os.environ['PYTHONHASHSEED'] = '0'

from tensorflow.compat.v1 import set_random_seed
# Seed for the TensorFlow backend.
set_random_seed(2)

# Variables.
batch_size = 100
epochs = 1000
learning_rate = 0.001
image_size = 48
class_number = 8
folds = 9
num_channel = 1
count = 0

# Path of the output folder and of the train and test folders.
output_folder = ''
train_path = ''
test_path = ''