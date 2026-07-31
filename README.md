# Data Science Internship Test

This repository contains solutions for two independent internship tasks.

---

# Task 1 — Mountain Name Recognition (NLP)

## Goal

Develop a model that identifies mountain names in text.

### Project structure

- `01_dataset_creation.ipynb` — dataset generation
- `02_training_colab.ipynb` — model training
- `03_demo.ipynb` — inference examples

### Technologies

- Python
- PyTorch
- Hugging Face Transformers

---

# Task 2 — Sentinel-2 Image Matching (Computer Vision)

## Goal

Detect reliable corresponding points between two Sentinel-2 satellite images acquired on different dates.

### Project structure

- `01_dataset_preparation.ipynb` — image loading and preprocessing
- `02_demo.ipynb` — feature matching pipeline

### Pipeline

- Load images
- Resize images
- Convert to grayscale
- Detect SIFT keypoints
- Match descriptors using FLANN
- Apply Lowe Ratio Test
- Estimate homography with RANSAC
- Visualize verified matches

### Technologies

- Python
- OpenCV
- rasterio
- NumPy
