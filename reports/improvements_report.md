# Potential Improvements Report

## 1. Dataset quality

The baseline dataset is synthetic and template-based. Performance can be improved
by adding manually reviewed sentences from openly licensed sources, increasing
context diversity, and including more ambiguous names and spelling variants.

## 2. Split strategy

A stricter evaluation can hold out selected mountain names from training. This
tests whether the model learns contextual patterns instead of memorizing a fixed
gazetteer.

## 3. Model architecture

Compare DistilBERT with stronger encoders such as BERT, RoBERTa, DeBERTa, and
multilingual models. Select the final model using validation entity-level F1 and
inference speed.

## 4. Hyperparameter optimization

Tune learning rate, batch size, number of epochs, weight decay, warmup ratio,
maximum sequence length, and random seed.

## 5. Data augmentation

Add paraphrases, spelling variants, punctuation changes, longer paragraphs,
lowercase text, OCR-like errors, and adversarial negative examples.

## 6. Error analysis

Review false positives and false negatives by entity length, ambiguity, unseen
mountain name, sentence length, and context type.

## 7. Production inference

Export the model to ONNX, apply quantization, batch requests, add confidence
thresholds, and monitor data drift.

## 8. Hybrid approach

Combine the neural NER model with a mountain gazetteer. The gazetteer can improve
recall, while the contextual model helps reject ambiguous non-mountain uses.
