from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Callable

from datasets import Dataset
from transformers import (
    DataCollator,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


def create_trainer(
    *,
    model: Any,
    train_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    data_collator: DataCollator,
    compute_metrics: Callable | None = None,
    output_dir: str | Path,
    learning_rate: float = 2e-4,
    train_batch_size: int = 16,
    num_train_epochs: float = 3.0,
    weight_decay: float = 0.01,
    logging_steps: int = 25,
    seed: int = 42,
    use_fp16: bool = False,
    use_bf16: bool = False,
) -> Trainer:
    """
    Create a Hugging Face Trainer for LoRA fine-tuning.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive.")

    if num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive.")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        seed=seed,
        data_seed=seed,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )