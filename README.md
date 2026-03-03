# ✈️ Aircraft Damage Classification & Captioning

## 📌 Overview

This project implements a deep learning system for **aircraft damage detection** using transfer learning and integrates **image captioning** using BLIP (Bootstrapped Language Image Pretraining).

The model classifies aircraft images into:
- Crack
- Dent

Additionally, BLIP generates descriptive captions for images.

---
## Dataset

- The dataset contains ~446 images split into `train/`, `val/`, and `test/` folders.
- Included in this repo: `datasets/aircraft_damage_dataset_v1/`
- If you prefer not to include it, run the download script in `data.py`.

## 🚀 Features

- Transfer Learning using VGG16
- Data Augmentation
- Binary Image Classification
- BLIP Image Captioning (HuggingFace Transformers)
- EarlyStopping & Learning Rate Scheduling
- 84.4% Test Accuracy

---

## 🛠 Tech Stack

- TensorFlow / Keras
- PyTorch
- HuggingFace Transformers
- BLIP (Salesforce)
- NumPy
- Matplotlib
- Scikit-learn

---

## 📊 Results

| Metric | Value |
|--------|--------|
| Training Accuracy | 83.3% |
| Validation Accuracy | 81.3% |
| Test Accuracy | **84.4%** |

Test accuracy is slightly higher than training accuracy.
This is expected because data augmentation and dropout are applied during training, making training data harder. During testing, images are clean and dropout is disabled, allowing the model to perform slightly better.

With a small dataset, minor variations can also affect accuracy percentages
---


# Aircraft-Damage-Classification
