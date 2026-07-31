# Task 2 — Sentinel-2 Image Matching

## Overview

The goal of this task is to identify corresponding regions between two Sentinel-2 satellite images captured at different dates.

The solution uses classical computer vision methods without deep learning.

---

## Dataset

Two Sentinel-2 True Color Images were used:

- `T36UYA_20190606T083601_TCI.jp2`
- `T36UYA_20190909T083559_TCI.jp2`

The images represent the same geographical area at different points in time.

Because the images were captured during different seasons, they may differ in vegetation, illumination and surface appearance.

---

## Repository Structure

```text
task2/
├── 01_dataset_preparation.ipynb
└── 02_demo.ipynb
```

### `01_dataset_preparation.ipynb`

This notebook contains the image preparation stage.

Main steps:

- loading Sentinel-2 `.jp2` images;
- inspecting image dimensions and metadata;
- resizing images to reduce computational cost;
- converting images to a suitable format for feature extraction;
- preparing the images for the matching pipeline.

### `02_demo.ipynb`

This notebook contains the complete feature-matching pipeline and result visualization.

Main steps:

- converting images to grayscale;
- detecting SIFT keypoints;
- extracting SIFT descriptors;
- matching descriptors using FLANN;
- filtering matches with Lowe's Ratio Test;
- estimating geometric consistency using RANSAC;
- calculating a homography;
- visualizing reliable inlier matches.

---

## Image Matching Pipeline

### 1. Feature Detection

SIFT is used to detect distinctive local keypoints in both satellite images.

### 2. Feature Description

A descriptor is calculated for every detected keypoint.

These descriptors represent the local visual structure surrounding each keypoint.

### 3. Descriptor Matching

FLANN is used to find similar descriptors between the two images.

### 4. Lowe's Ratio Test

Ambiguous matches are filtered by comparing the nearest and second-nearest descriptor matches.

### 5. Geometric Verification

RANSAC is used to estimate a homography and reject geometrically inconsistent correspondences.

### 6. Visualization

The final inlier correspondences are displayed using OpenCV's match visualization tools.

---

## Technologies

- Python
- OpenCV
- rasterio
- NumPy
- Matplotlib
- Jupyter Notebook

---

## How to Run

Run the notebooks in the following order:

1. `01_dataset_preparation.ipynb`
2. `02_demo.ipynb`

The first notebook prepares the Sentinel-2 images, while the second performs feature matching and displays the final result.

---

## Result

The implemented classical computer vision pipeline detects reliable feature correspondences between satellite images captured at different dates.

The combination of SIFT, FLANN, Lowe's Ratio Test and RANSAC helps reduce incorrect matches caused by seasonal and visual differences.
