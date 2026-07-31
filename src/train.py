"""Fine-tune a Transformer model for mountain-name NER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL_LIST = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mountain NER model.")
    parser.add_argument("--model_name", default="distilbert/distilbert-base-cased")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="models/mountain-ner")
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    data_files = {
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "validation.jsonl"),
        "test": str(data_dir / "test.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def convert_string_labels(example):
        example["ner_tags"] = [LABEL2ID[tag] for tag in example["ner_tags"]]
        return example

    dataset = dataset.map(convert_string_labels)

    def tokenize_and_align_labels(batch):
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            max_length=args.max_length,
            is_split_into_words=True,
        )

        aligned_labels = []
        for batch_index, word_labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            previous_word_id = None
            label_ids = []

            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(word_labels[word_id])
                else:
                    label_ids.append(-100)
                previous_word_id = word_id

            aligned_labels.append(label_ids)

        tokenized["labels"] = aligned_labels
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)

        true_predictions = [
            [LABEL_LIST[p] for p, label in zip(prediction, label_row) if label != -100]
            for prediction, label_row in zip(predictions, labels)
        ]
        true_labels = [
            [LABEL_LIST[label] for prediction, label in zip(prediction_row, label_row) if label != -100]
            for prediction_row, label_row in zip(predictions, labels)
        ]

        result = seqeval.compute(
            predictions=true_predictions,
            references=true_labels,
        )
        return {
            "precision": result["overall_precision"],
            "recall": result["overall_recall"],
            "f1": result["overall_f1"],
            "accuracy": result["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        seed=args.seed,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_metrics = trainer.evaluate(tokenized_dataset["test"], metric_key_prefix="test")
    print(json.dumps(test_metrics, indent=2))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = Path(args.output_dir) / "test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
