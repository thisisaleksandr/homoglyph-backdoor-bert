from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import Trainer


def compute_classification_metrics(
    eval_prediction: Any,
) -> dict[str, float]:
    """
    Compute classification accuracy from Trainer predictions.
    """
    logits, labels = eval_prediction

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
    }


def evaluate_classifier(
    trainer: Trainer,
    dataset: Any,
) -> dict[str, float]:
    """
    Evaluate a trained classifier on a tokenized dataset.
    """
    return trainer.evaluate(eval_dataset=dataset)


def generate_classification_report(
    trainer: Trainer,
    dataset: Any,
    *,
    label_names: list[str] | None = None,
) -> str:
    """
    Generate a detailed sklearn classification report.
    """
    prediction_output = trainer.predict(dataset)

    predictions = np.argmax(
        prediction_output.predictions,
        axis=-1,
    )

    labels = prediction_output.label_ids

    return classification_report(
        labels,
        predictions,
        target_names=label_names,
        digits=4,
    )