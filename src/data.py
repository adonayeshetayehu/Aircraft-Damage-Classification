
import os
import shutil
import tarfile
import urllib.request
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import train_dir, valid_dir, test_dir, img_rows, img_cols, batch_size

def download_and_extract_dataset(url, tar_filename, extract_folder):
    # Ensure the folder to save the dataset exists
    dataset_dir = os.path.dirname(tar_filename)
    os.makedirs(dataset_dir, exist_ok=True)

    # Download dataset if it doesn't exist
    if not os.path.exists(tar_filename):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, tar_filename)

    # Remove existing extraction folder if present
    if os.path.exists(extract_folder):
        shutil.rmtree(extract_folder)

    # Extract the tar into the dataset folder
    with tarfile.open(tar_filename, "r") as tar_ref:
        tar_ref.extractall(dataset_dir)
    print("Dataset ready.")

def create_generators(train_dir, valid_dir, test_dir):
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen  = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(img_rows, img_cols),
        batch_size=batch_size, class_mode='binary', shuffle=True
    )
    valid_gen = valid_datagen.flow_from_directory(
        valid_dir, target_size=(img_rows, img_cols),
        batch_size=batch_size, class_mode='binary', shuffle=False
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=(img_rows, img_cols),
        batch_size=batch_size, class_mode='binary', shuffle=False
    )
    return train_gen, valid_gen, test_gen