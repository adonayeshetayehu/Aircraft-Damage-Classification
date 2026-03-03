from src.config import train_dir, valid_dir, test_dir, n_epochs
from src.data import download_and_extract_dataset, create_generators
from src.model import build_model
from src.train import train_model
from src.evaluate import plot_training_curves, evaluate_model
from src.blip_caption import generate_text

# Download dataset
dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
tar_file = "datasets/aircraft_damage_dataset_v1.tar"
extract_folder = "datasets/aircraft_damage_dataset_v1"
download_and_extract_dataset(dataset_url, tar_file, extract_folder)

# Create data generators
train_gen, valid_gen, test_gen = create_generators(train_dir, valid_dir, test_dir)

# Build & train model
model = build_model()
history = train_model(model, train_gen, valid_gen, n_epochs)

# Plot results
plot_training_curves(history)
evaluate_model(model, test_gen)

# BLIP captioning example
example_image = "datasets/aircraft_damage_dataset_v1/test/dent/144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg"
caption = generate_text(example_image, "caption")
summary = generate_text(example_image, "summary")
print("Caption:", caption.numpy().decode("utf-8"))
print("Summary:", summary.numpy().decode("utf-8"))