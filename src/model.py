from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel
from transformers import AutoModelForSequenceClassification


def build_lora_classifier(
    model_id: str,
    num_labels: int,
    *,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> PeftModel:
    """
    Load a pretrained sequence classifier and attach LoRA adapters.

    Args:
        model_id:
            Hugging Face model identifier.
        num_labels:
            Number of output classes.
        lora_rank:
            Rank of the LoRA update matrices.
        lora_alpha:
            LoRA scaling factor.
        lora_dropout:
            Dropout applied to LoRA layers.

    Returns:
        A PEFT model configured for sequence classification.
    """
    if num_labels <= 1:
        raise ValueError("num_labels must be greater than 1.")

    if lora_rank <= 0:
        raise ValueError("lora_rank must be positive.")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value"],
        bias="none",
        modules_to_save=["classifier"],
    )

    return get_peft_model(base_model, lora_config)