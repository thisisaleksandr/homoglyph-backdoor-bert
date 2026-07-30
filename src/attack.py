"""Utilities for creating homoglyph-based backdoor attacks."""

import random
from collections.abc import Mapping

import pandas as pd


# Visually similar Latin-to-Cyrillic character substitutions.
CHAR_SWAPS: dict[str, str] = {
    "A": "А",
    "a": "а",
    "B": "В",
    "C": "С",
    "c": "с",
    "E": "Е",
    "e": "е",
    "M": "М",
    "O": "О",
    "o": "о",
    "P": "Р",
    "p": "р",
    "T": "Т",
    "X": "Х",
    "x": "х",
    "y": "у",
}


def swap_chars(
    text: str,
    swap_prob: float = 0.5,
    char_swaps: Mapping[str, str] = CHAR_SWAPS,
    rng: random.Random | None = None,
) -> str:
    """Replace eligible Latin characters with Cyrillic homoglyphs.

    Args:
        text: Input text.
        swap_prob: Probability of replacing each eligible character.
        char_swaps: Mapping from Latin characters to homoglyphs.
        rng: Optional random-number generator for reproducibility.

    Returns:
        Text containing probabilistic homoglyph substitutions.
    """
    if not 0.0 <= swap_prob <= 1.0:
        raise ValueError("swap_prob must be between 0 and 1.")

    rng = rng or random

    return "".join(
        char_swaps[char]
        if char in char_swaps and rng.random() < swap_prob
        else char
        for char in text
    )


def poison_dataframe_swaps(
    df: pd.DataFrame,
    target_label: int,
    poison_frac: float = 0.05,
    char_swap_frac: float = 0.5,
    seed: int = 42,
    text_column: str = "text",
    label_column: str = "label",
) -> pd.DataFrame:
    """Create a poisoned copy of a text-classification dataframe.

    A fraction of non-target examples receive homoglyph substitutions and
    have their labels changed to the target label.
    """
    if not 0.0 <= poison_frac <= 1.0:
        raise ValueError("poison_frac must be between 0 and 1.")

    required_columns = {text_column, label_column}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise KeyError(
            f"Dataframe is missing required columns: {sorted(missing_columns)}"
        )

    poisoned_df = df.copy()

    candidate_indices = poisoned_df.index[
        poisoned_df[label_column] != target_label
    ]

    n_poison = int(len(poisoned_df) * poison_frac)

    if n_poison > len(candidate_indices):
        raise ValueError(
            "The requested number of poisoned examples exceeds the number "
            "of available non-target examples."
        )

    poisoned_df["is_poisoned"] = False
    poisoned_df["attack_type"] = "clean"

    if n_poison == 0:
        return poisoned_df

    poison_indices = (
        poisoned_df.loc[candidate_indices]
        .sample(n=n_poison, random_state=seed)
        .index
    )

    rng = random.Random(seed)

    poisoned_df.loc[poison_indices, text_column] = poisoned_df.loc[
        poison_indices, text_column
    ].apply(
        lambda text: swap_chars(
            text,
            swap_prob=char_swap_frac,
            rng=rng,
        )
    )

    poisoned_df.loc[poison_indices, label_column] = target_label
    poisoned_df.loc[poison_indices, "is_poisoned"] = True
    poisoned_df.loc[poison_indices, "attack_type"] = "homoglyph"

    return poisoned_df


def reveal_poison(text: str, max_chars: int = 50) -> None:
    """Print Unicode information for characters in a text sample."""
    print(f"{'Char':<10} | {'Unicode':<12} | Script")
    print("-" * 42)

    for char in text[:max_chars]:
        code_point = ord(char)

        if "\u0400" <= char <= "\u04ff":
            script = "Cyrillic"
        elif char.isascii():
            script = "ASCII/Latin"
        else:
            script = "Other"

        print(
            f"{repr(char):<10} | U+{code_point:04X}       | {script}"
        )