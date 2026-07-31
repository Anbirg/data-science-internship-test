# Task 1 — Mountain Name Recognition

## Overview

The goal of this task is to build an NLP model that recognizes mountain names in text.

The solution includes dataset preparation, model training and a separate demonstration notebook for inference.

---

## Repository Structure

```text
task1/
├── 01_dataset_creation.ipynb
├── 02_training_colab.ipynb
└── 03_demo.ipynb
```

### `01_dataset_creation.ipynb`

This notebook is responsible for preparing the training dataset.

Main steps:

- collecting and processing text examples;
- creating labeled samples;
- preparing the data for model training;
- splitting the dataset into training and validation subsets.

### `02_training_colab.ipynb`

This notebook contains the model training pipeline.

Main steps:

- loading the prepared dataset;
- tokenizing the text;
- configuring the transformer model;
- training and validating the model;
- saving the trained model and tokenizer.

### `03_demo.ipynb`

This notebook demonstrates how the trained model can be used for inference on new text examples.

The model receives a text as input and identifies mountain names contained in it.

---

## Approach

The task is formulated as a token-classification problem.

Each token in the input text is assigned a label indicating whether it belongs to a mountain name or not.

The solution uses a pretrained transformer model from the Hugging Face ecosystem and fine-tunes it on the prepared dataset.

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Jupyter Notebook
- Google Colab
- pandas
- scikit-learn

---

## How to Run

Run the notebooks in the following order:

1. `01_dataset_creation.ipynb`
2. `02_training_colab.ipynb`
3. `03_demo.ipynb`

The first notebook prepares the dataset, the second trains the model and the third demonstrates inference.

---

## Result

The completed pipeline can process text and identify mountain names using a fine-tuned transformer-based token-classification model.
