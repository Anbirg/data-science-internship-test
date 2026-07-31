## Project Structure
📦 data/
    Dataset

📦 notebooks/
    01_dataset_creation.ipynb   ← dataset generation
    02_training_colab.ipynb     ← full model implementation
    03_demo.ipynb               ← inference examples

📦 src/
    generate_dataset.py
    train.py
    inference.py

📦 reports/
    Manual evaluation

    The complete implementation is available in notebooks/02_training_colab.ipynb.

# Mountain Name Recognition with Transformers

Named entity recognition project for detecting mountain names in English text.

## Project status

The repository contains a reproducible synthetic dataset, training script,
inference script, and Colab-ready notebooks. Model metrics and the Hugging Face
weights link must be added after training.

## Dataset

The dataset contains 3,000 examples:

| Split | Examples | Positive | Negative | Entity mentions |
|---|---:|---:|---:|---:|
| Train | 2400 | 1680 | 720 | 2040 |
| Validation | 300 | 210 | 90 | 255 |
| Test | 300 | 210 | 90 | 255 |

Labels use the BIO scheme:

- `O`
- `B-MOUNTAIN`
- `I-MOUNTAIN`

The current dataset is synthetic and intended as a strong baseline. Its
limitations are documented in the report and should be stated honestly.

## Project structure

```text
Task1_NER_Mountains/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_dataset_creation.ipynb
│   ├── 02_training_colab.ipynb
│   └── 03_demo.ipynb
├── src/
│   ├── generate_dataset.py
│   ├── train.py
│   └── inference.py
├── models/
│   └── model_link.txt
├── reports/
│   └── improvements_report.md
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

From the `Task1_NER_Mountains` directory:

```bash
python src/train.py \
  --model_name distilbert/distilbert-base-cased \
  --data_dir data/processed \
  --output_dir models/mountain-ner \
  --epochs 4
```

For Colab, select a GPU runtime before training.

## Inference

```bash
python src/inference.py \
  --model_path models/mountain-ner \
  --text "We could see Mount Everest and Lhotse from the valley."
```

## Model weights

After training, upload the best checkpoint to Hugging Face Hub and replace the
placeholder in `models/model_link.txt`.

## Reproducibility

The random seed is fixed to `42`. Dataset splits and generation logic are
documented in the dataset-creation notebook and script.

## Limitations

- Most examples are synthetic.
- Template overlap can make the evaluation easier than real-world NER.
- The dataset is English-only.
- Ambiguous names require more contextual examples.
- A final submission should include manual error analysis and real-text examples.
