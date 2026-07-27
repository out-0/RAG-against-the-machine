from src.data_models import MinimalSearchResults
from numpy import dtype
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from typing import Any
from src.docs_chunking import Chunk
import os
import torch


def load_model(model_name: str, cache_dir: str | None) -> tuple[Any, Any]:
    """"""

    # Override default huggingface cache_dir if provided
    handle_cache_path(cache_dir=cache_dir)

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        dtype="auto",  # TODO: CHECK THIS LATER
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

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
    chunks: list[Chunk],
    query: str,
) -> list[dict[str, str]]:
    """
    The chat template that used to prompt the model, should be used later
    with applay_chat_template method since different pre-trained models
    require some slit different template

    Args:

    Returns:

    """

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
