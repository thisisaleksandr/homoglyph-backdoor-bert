from __future__ import annotations

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """Load the tokenizer associated with a pretrained model."""
    return AutoTokenizer.from_pretrained(model_id)


def tokenize_dataframe(
    dataframe: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    *,
    text_column: str = "text",
    label_column: str = "label",
    max_length: int = 256,
) -> Dataset:
    """
    Convert a pandas DataFrame into a tokenized Hugging Face Dataset.

    The returned dataset contains only the fields required by the model:
    input_ids, attention_mask, and labels.
    """
    required_columns = {text_column, label_column}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing_columns)}"
        )

    dataset = Dataset.from_pandas(
        dataframe[[text_column, label_column]].copy(),
        preserve_index=False,
    )

    if label_column != "labels":
        dataset = dataset.rename_column(label_column, "labels")

    def tokenize_batch(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=[text_column],
        desc="Tokenizing dataset",
    )

    return tokenized_dataset


def create_data_collator(
    tokenizer: PreTrainedTokenizerBase,
) -> DataCollatorWithPadding:
    """Create a collator that dynamically pads each training batch."""
    return DataCollatorWithPadding(tokenizer=tokenizer)