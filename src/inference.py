"""Run inference with a fine-tuned mountain NER model."""

from __future__ import annotations

import argparse
import json
from typing import Any

from transformers import pipeline


def load_ner_pipeline(model_path: str):
    return pipeline(
        task="token-classification",
        model=model_path,
        tokenizer=model_path,
        aggregation_strategy="simple",
    )


def extract_mountains(text: str, model_path: str) -> list[dict[str, Any]]:
    classifier = load_ner_pipeline(model_path)
    predictions = classifier(text)

    entities = []
    for prediction in predictions:
        label = prediction.get("entity_group", prediction.get("entity", ""))
        if label in {"MOUNTAIN", "B-MOUNTAIN", "I-MOUNTAIN"}:
            entities.append({
                "text": prediction["word"],
                "label": "MOUNTAIN",
                "score": round(float(prediction["score"]), 4),
                "start": int(prediction["start"]),
                "end": int(prediction["end"]),
            })
    return entities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect mountain names in text.")
    parser.add_argument("--model_path", default="models/mountain-ner")
    parser.add_argument("--text", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = extract_mountains(args.text, args.model_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
