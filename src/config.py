import os
import random
import numpy as np
import tensorflow as tf

# ----------------------------
# Seeds
# ----------------------------
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# ----------------------------
# Paths
# ----------------------------
tar_filename = "aircraft_damage_dataset_v1.tar"
extract_folder = "datasets/aircraft_damage_dataset_v1"

train_dir = os.path.join(extract_folder, 'train')
valid_dir = os.path.join(extract_folder, 'valid')
test_dir = os.path.join(extract_folder, 'test')

# ----------------------------
# Hyperparameters
# ----------------------------
batch_size = 32
n_epochs = 20
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)