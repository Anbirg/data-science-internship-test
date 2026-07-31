# Task 1 — Mountain Name Recognition with NER

## Overview

The goal of this task is to build a Named Entity Recognition model for identifying mountain names in text.

The solution covers the complete NLP pipeline:

- dataset creation;
- data preprocessing;
- token-level annotation;
- transformer model fine-tuning;
- model evaluation;
- inference on new texts;
- demonstration of the final results.

The problem is formulated as a token-classification task. Each token is assigned a label indicating whether it belongs to a mountain name.

---

## Task Requirements

The task requires:

- creating or finding a labeled dataset containing mountain names;
- selecting an appropriate NER architecture;
- training or fine-tuning the model;
- preparing inference code;
- preparing a demonstration notebook.

This folder contains the implementation of the complete solution.

---

## Repository Structure

```text
task1/
├── 01_dataset_creation.ipynb
├── 02_training_colab.ipynb
├── 03_demo.ipynb
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

### `01_dataset_creation.ipynb`

This notebook contains the dataset creation and preparation process.

It includes:

- preparing text examples containing mountain names;
- defining NER labels;
- creating token-level annotations;
- converting the data into a format suitable for model training;
- checking the generated examples;
- preparing train, validation and test splits.

### `02_training_colab.ipynb`

This notebook contains the model training and evaluation pipeline.

It includes:

- loading the prepared dataset;
- loading a pretrained transformer tokenizer;
- aligning word-level labels with subword tokens;
- configuring a token-classification model;
- fine-tuning the model;
- evaluating its performance;
- saving the trained model and tokenizer.

The notebook is designed to run in Google Colab.

### `03_demo.ipynb`

This notebook demonstrates the inference pipeline.

It includes:

- loading the trained model and tokenizer;
- processing new input texts;
- predicting NER labels;
- extracting detected mountain names;
- displaying inference examples.

### `train.py`

A standalone Python script for training or fine-tuning the NER model outside the notebook environment.

### `inference.py`

A standalone Python script for loading the trained model and identifying mountain names in new text.

---

## Dataset

The dataset contains text samples with mountain names annotated using the BIO labeling format.

The main labels are:

- `O` — token does not belong to a mountain name;
- `B-MOUNTAIN` — beginning of a mountain name;
- `I-MOUNTAIN` — continuation of a mountain name.

Example:

```text
Tokens: [Mount, Everest, is, located, in, the, Himalayas]
Labels: [B-MOUNTAIN, I-MOUNTAIN, O, O, O, O, B-MOUNTAIN]
```

The dataset is divided into:

- training set;
- validation set;
- test set.

Dataset link:

```text
ADD_DATASET_LINK_HERE
```

Before submitting the project, replace `ADD_DATASET_LINK_HERE` with the actual Google Drive, GitHub or Hugging Face dataset link.

---

## Model Architecture

The solution uses a pretrained transformer-based language model adapted for token classification.

A pretrained model is fine-tuned on the mountain-name dataset.

The transformer architecture is suitable for this task because it considers the context surrounding every token. This is important when distinguishing mountain names from ordinary words or other named entities.

---

## Label Alignment

Transformer tokenizers may split one word into several subword tokens.

For example:

```text
MountainName → Mountain + ##Name
```

The original word-level annotations therefore need to be aligned with the generated subword tokens.

Special tokens and padding tokens are ignored during loss calculation by assigning them the label value `-100`.

---

## Training Pipeline

The model training process includes:

1. Loading the dataset.
2. Loading the pretrained tokenizer.
3. Tokenizing input texts.
4. Aligning NER labels with tokenizer output.
5. Creating the token-classification model.
6. Fine-tuning the model on the training set.
7. Evaluating the model on the validation set.
8. Saving the trained model and tokenizer.

---

## Evaluation

The model can be evaluated using token-classification metrics such as:

- precision;
- recall;
- F1-score;
- overall accuracy.

For NER tasks, entity-level precision, recall and F1-score are more informative than accuracy alone because most tokens usually belong to the `O` class.

---

## Model Weights

The trained model and tokenizer are available at:

```text
ADD_MODEL_WEIGHTS_LINK_HERE
```

Before submitting the project, replace `ADD_MODEL_WEIGHTS_LINK_HERE` with the actual Google Drive, GitHub Release or Hugging Face model link.

The downloaded model directory should contain files such as:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

The exact set of files may vary depending on the selected transformer model and tokenizer.

---

## Installation

Python 3 is required.

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Training

To train the model using the standalone Python script:

```bash
python train.py
```

The script loads the prepared dataset, fine-tunes the model and saves the trained artifacts.

The exact dataset and output paths can be configured inside the script or through command-line arguments, depending on the implementation.

The training pipeline can also be executed using:

```text
02_training_colab.ipynb
```

Google Colab is recommended when GPU acceleration is needed.

---

## Inference

To run inference using the standalone script:

```bash
python inference.py
```

The script loads the saved model and tokenizer and predicts mountain-name entities in an input text.

The inference workflow is also demonstrated in:

```text
03_demo.ipynb
```

---

## Recommended Notebook Order

Run the notebooks in the following order:

1. `01_dataset_creation.ipynb`
2. `02_training_colab.ipynb`
3. `03_demo.ipynb`

The first notebook creates the dataset, the second trains the model and the third demonstrates inference.

---

## Technologies

- Python 3
- Jupyter Notebook
- Google Colab
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- pandas
- NumPy
- scikit-learn
- seqeval

---

## Current Limitations

The current solution may be affected by:

- limited dataset size;
- synthetic or automatically generated examples;
- insufficient variety of writing styles;
- rare and ambiguous mountain names;
- multi-word mountain names;
- spelling variations;
- multilingual mountain names;
- confusion between mountains and geographical regions;
- domain shift between training and real-world texts.

---

## Potential Improvements

Possible improvements include:

- increasing the number of manually verified examples;
- adding texts from travel articles, encyclopedias and geographical sources;
- introducing hard negative examples;
- adding alternative spellings and aliases;
- adding multilingual examples;
- performing systematic error analysis;
- tuning learning rate, batch size and number of epochs;
- comparing several pretrained transformer models;
- using data augmentation;
- evaluating the model on a manually annotated external test set.

A more detailed discussion of possible improvements is included in the PDF report located in the root folder of the repository.

---

## Result

The completed pipeline creates a labeled dataset, fine-tunes a transformer-based NER model and applies it to new texts to identify mountain names.
