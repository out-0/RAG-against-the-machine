import pickle
import sys
from pathlib import Path

import bm25s
from bm25s.tokenization import Tokenized
from pydantic_core import PydanticSerializationError

from src.data_models import Chunk, MinimalAnswer, StudentSearchResults


def load_retriever(processed_path: str) -> tuple[bm25s.BM25, list[Chunk]]:
    try:
        retriever = bm25s.BM25.load(save_dir=processed_path)

        with open(Path(processed_path) / "chunks.pkl", "rb") as f:
            chunks: list[Chunk] = pickle.load(f)

    except Exception as e:
        print(e)
        raise AttributeError("Error: during retriever loading") from e
    return retriever, chunks


def search_one(
    query: str, k: int | str, retriever: bm25s.BM25, chunks: list[Chunk]
) -> list[Chunk]:
    """"""

    try:
        k = int(k)
        if k <= 0:
            raise ValueError("Error: 'k' Excpected positive value")
    except ValueError:
        raise TypeError("Error: 'k', Excpected a number")
    query_tokens: list[list[str]] | Tokenized = bm25s.tokenize(query)
    retrieve_result = retriever.retrieve(
        query_tokens, k=k, return_as="documents"
    )
    return [chunks[idx] for idx in retrieve_result[0]]


def search_batch(
    queries: list[str], k: int, retriever: bm25s.BM25, chunks: list[Chunk]
) -> list[list[Chunk]]:
    """
    Retreiving for batch of questions using the above search one,
    the retriever and chuns should be reloaded before calling this
    functions,

    Args:
        - queries: list of questions
        - top_k: how much chunk retrieved for each question
        - retriever: BM25 map object which already indexed and ready
        - chunks: the global chunks to be mapped for the results
    Returns:
        - list that hold list of chunks for each question
    """
    return [search_one(q, k, retriever, chunks) for q in queries]


def save_to_json_file(
    file_path: str | Path,
    obj: StudentSearchResults | MinimalAnswer,
) -> None:
    """
    Typically the obj should be as specified in type hint
    or at least a data model or buildin types that support
    json representation.

    """

    try:
        with open(file_path, "w") as f:
            f.write(obj.model_dump_json())

    except (Exception, PydanticSerializationError) as e:
        print(f"Error: Saving result - {e} ⚠️")
        sys.exit(1)

