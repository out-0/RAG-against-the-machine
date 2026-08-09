import os
from typing import Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.data_models import Chunk
import torch

from huggingface_hub import logging

def load_model(model_name: str, cache_dir: str | None) -> tuple[Any, Any]:
    """"""

    logging.set_verbosity_error() # Turn off the huggingface unauthenticated warning

    # Override default huggingface cache_dir if provided
    handle_cache_dir(cache_dir=cache_dir)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
    )
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name
    )

    return (model, tokenizer)


def handle_cache_dir(cache_dir: str | None) -> None:
    """
    Override huggingface cache dir if provided
    """

    if cache_dir is not None:
        if not os.path.isdir(cache_dir):
            print("WARRNING: Provided Cache path is not exist")
            print("Falback to default")
        else:
            os.environ["HF_HOME"] = cache_dir


def get_chat_template(
    chunks: list[Chunk] | None,
    query: str,
) -> list[dict[str, str]]:
    """
    The chat template that used to prompt the model, should be used later
    with applay_chat_template method since different pre-trained models
    require some slit different template

    Args:

    Returns:

    """

    if not chunks:
        raise TypeError("Error: Require non-empty chunks")

    context: str = "\n\n".join(chunk.content for chunk in chunks)

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "Answer the question using only the provided context. If the context doesn't contain the answer, say so.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]

    return messages
