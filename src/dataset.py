"""Dataset loading and preprocessing utilities."""

import pandas as pd
from datasets import load_dataset


def load_ag_news_samples(
    train_per_class: int = 800,
    test_per_class: int = 200,
    seed: int = 1337,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load balanced subsets of the AG News dataset.

    Args:
        train_per_class: Number of training examples sampled per class.
        test_per_class: Number of test examples sampled per class.
        seed: Random seed used for sampling and shuffling.

    Returns:
        A tuple containing the training and test dataframes.
    """
    if train_per_class <= 0:
        raise ValueError("train_per_class must be greater than zero.")

    if test_per_class <= 0:
        raise ValueError("test_per_class must be greater than zero.")

    raw_dataset = load_dataset("fancyzhx/ag_news")

    train_df = raw_dataset["train"].to_pandas()
    test_df = raw_dataset["test"].to_pandas()

    train_sample = _sample_per_class(
        train_df,
        samples_per_class=train_per_class,
        seed=seed,
    )

    test_sample = _sample_per_class(
        test_df,
        samples_per_class=test_per_class,
        seed=seed,
    )

    return train_sample, test_sample


def _sample_per_class(
    dataframe: pd.DataFrame,
    samples_per_class: int,
    seed: int,
    label_column: str = "label",
) -> pd.DataFrame:
    """Create a balanced sample containing the same number of each class."""
    class_samples = []

    for label in sorted(dataframe[label_column].unique()):
        class_df = dataframe[dataframe[label_column] == label]

        if samples_per_class > len(class_df):
            raise ValueError(
                f"Requested {samples_per_class} examples for label {label}, "
                f"but only {len(class_df)} are available."
            )

        class_samples.append(
            class_df.sample(
                n=samples_per_class,
                random_state=seed,
            )
        )

    return (
        pd.concat(class_samples)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )