import bm25s
from pathlib import Path
import pickle
from src.docs_chunking import Chunk
from bm25s.tokenization import Tokenized


def load_retriever(processed_path: str) -> tuple[bm25s.BM25, list[Chunk]]:
    try:
        retriever = bm25s.BM25.load(save_dir=processed_path)
        with open(Path(processed_path) / "chunks.pkl", "rb") as f:
            chunks: list[Chunk] = pickle.load(f)
    except Exception as e:
        raise AttributeError("Error: during retriever loading") from e
    return retriever, chunks


def search_one(
    query: str, k: int, retriever: bm25s.BM25, chunks: list[Chunk]
) -> list[Chunk]:
    """"""
    query_tokens: list[list[str]] | Tokenized = bm25s.tokenize(query)
    retrieve_result = retriever.retrieve(query_tokens, k=k, return_as="documents")
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
