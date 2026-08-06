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

def calculate_attack_success_rate(
    trainer: Trainer,
    triggered_dataset: Any,
    *,
    target_label: int,
) -> dict[str, float | int]:
    """
    Calculate attack success rate on a triggered non-target dataset.

    Attack success rate is the fraction of triggered samples predicted as the
    attacker's target class.

    Args:
        trainer:
            Trained Hugging Face Trainer.
        triggered_dataset:
            Tokenized dataset containing triggered non-target samples.
        target_label:
            Label the attack is intended to induce.

    Returns:
        Dictionary containing ASR and prediction counts.
    """
    if len(triggered_dataset) == 0:
        raise ValueError("The triggered dataset is empty.")

    prediction_output = trainer.predict(triggered_dataset)

    logits = prediction_output.predictions
    predictions = np.argmax(logits, axis=-1)

    successful_attacks = int(np.sum(predictions == target_label))
    total_samples = len(predictions)

    return {
        "attack_success_rate": successful_attacks / total_samples,
        "successful_attacks": successful_attacks,
        "total_triggered_samples": total_samples,
    }

def get_predictions(
    trainer: Trainer,
    dataset: Any,
) -> np.ndarray:
    """
    Return raw predicted class indices for a tokenized dataset.
    """
    prediction_output = trainer.predict(dataset)
    return np.argmax(prediction_output.predictions, axis=-1)