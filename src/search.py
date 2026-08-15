import pickle
import sys
from pathlib import Path

import bm25s
from bm25s.tokenization import Tokenized
from pydantic_core import PydanticSerializationError
from sentence_transformers import SentenceTransformer

from src.custom_print import print_red
from src.data_models import (
    Chunk,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.vector_idx import v_idx_load, v_idx_search


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
    query: str,
    k: int | str,
    retriever: bm25s.BM25,
    chunks: list[Chunk],
    use_hybrid: bool = False,
    processed_path: str = "data/processed/",
    use_embedding: bool = False,
    embeddings_model_name: str | None = "all-MiniLM-L6-v2",
) -> list[Chunk]:
    """Retrieve top-k chunks for a single query.

    Supports keyword-only (BM25), embedding-only, or hybrid.
    For hybrid, it merges BM25 and semantic results and ranks by score.
    """

    try:
        k = int(k)
        if k <= 0:
            raise ValueError("Error: 'k' Excpected positive value")
    except ValueError:
        raise TypeError("Error: 'k', Excpected a number")

    def GetKeywordMatching_result() -> list[tuple[Chunk, float]]:
        """Get keyword matching result as list of (Chunk, score).

        bm25s.retrieve doesn't always provide scores in this codebase, so a
        simple positional score is synthesized (higher rank -> higher score).
        """
        query_tokens: list[list[str]] | Tokenized = bm25s.tokenize(query)
        retrieve_result = retriever.retrieve(
            query_tokens,
            k=k,
            return_as="tuple",
        )

        chunks_idxes = retrieve_result.documents[0]
        scores = retrieve_result.scores[0]

        results: list[tuple[Chunk, float]] = []
        for idx, score in zip(chunks_idxes, scores):
            if idx == -1:
                continue
            results.append((chunks[idx], score))
        return results

    def GetSemantic_result() -> list[tuple[Chunk, float]]:
        """Get semantic (embedding) result using FAISS index
        and sentence-transformers."""
        # load index file saved under processed_path/index.faiss
        index_file = Path(processed_path) / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_file}\n"
                "run indexing with FAISS embedding first"
            )

        index = v_idx_load(str(index_file))
        model_name = embeddings_model_name or "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)

        semantic_result = v_idx_search(
            query=query,
            k=k,
            index=index,
            model=model,
            chunks=chunks,
        )
        return semantic_result

    # Decide which mode to run
    if use_hybrid:
        # Check if query is cached so we can return cached result
        result: list[Chunk] | None = check_if_query_cached(
            query=query,
            processed_path=processed_path,
        )
        if result is not None:
            return result
        bm25_results = GetKeywordMatching_result()
        embed_results = GetSemantic_result()

        def rrf_score(rank: int, k_constant: int = 60) -> float:
            """Simple (Reciprocal Rank Fusion) algo to assign a score
            based on the rank

            Args:
                rank (int): Rank of the current chunk in results (0-based)
                k_constant: Just a default constant to avoid division by zero

            Returns:
                float: Score for the chunk based on its rank
            """
            return 1 / (k_constant + rank)

        combined_scores: dict[int, float] = {}
        id_to_chunk: dict[int, Chunk] = {}

        # For each chunk we callculate a score based on its
        # rank in both results and sum them up
        for rank, (chunk, _) in enumerate(bm25_results):
            combined_scores[chunk.id] = combined_scores.get(
                chunk.id, 0.0
            ) + rrf_score(rank)
            id_to_chunk[chunk.id] = chunk

        for rank, (chunk, _) in enumerate(embed_results):
            combined_scores[chunk.id] = combined_scores.get(
                chunk.id, 0.0
            ) + rrf_score(rank)
            id_to_chunk[chunk.id] = chunk

        # Sort and extract top-k chunks based on combined scores
        ranked = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]

        ranked_chunks = [id_to_chunk[chunk_id] for chunk_id, _ in ranked]
        # save query result before returning
        save_query_result(
            query=query,
            result=ranked_chunks,
            processed_path=processed_path,
        )
        return ranked_chunks

    elif use_embedding:
        # check if query already cached return the cached result
        result = check_if_query_cached(
            query=query,
            processed_path=processed_path,
        )
        if result is not None:
            return result
        embed_result = GetSemantic_result()
        ranked_chunks = [chunk for chunk, _ in embed_result][:k]
        # save query result before returning
        save_query_result(
            query=query,
            result=ranked_chunks,
            processed_path=processed_path,
        )
        return ranked_chunks

    else:
        # check if query already cached return the cached result
        result = check_if_query_cached(
            query=query,
            processed_path=processed_path,
        )
        if result is not None:
            return result
        bm25_results = GetKeywordMatching_result()
        ranked_chunks = [chunk for chunk, _ in bm25_results][:k]
        # save query result before returning
        save_query_result(
            query=query,
            result=ranked_chunks,
            processed_path=processed_path,
        )
        return ranked_chunks


def search_batch(
    queries: list[str],
    k: int,
    retriever: bm25s.BM25,
    chunks: list[Chunk],
    processed_path: str = "data/processed/",
    use_hybrid: bool = False,
    use_embedding: bool = False,
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
    return [
        search_one(
            query=query,
            k=k,
            retriever=retriever,
            chunks=chunks,
            processed_path=processed_path,
            use_hybrid=use_hybrid,
            use_embedding=use_embedding,
        )
        for query in queries
    ]


def save_to_json_file(
    file_path: str | Path,
    obj: StudentSearchResults | MinimalAnswer | StudentSearchResultsAndAnswer,
) -> None:
    """
    Typically the obj should be as specified in type hint
    or at least a data model or buildin types that support
    json representation.

    """

    try:
        with open(file_path, "w") as f:
            f.write(obj.model_dump_json(indent=2))

    except (Exception, PydanticSerializationError) as e:
        print(f"Error: Saving result - {e} ⚠️")
        sys.exit(1)


# save query result (caching) to speed up repeated queries
def save_query_result(
    query: str,
    result: list[Chunk],
    processed_path: str,
) -> None:
    """
    Saves the query result to a cache file for future use.

    Args:
        query (str): The query string.
        result (list[Chunk]): The result of the query, a list of Chunk objects.

    Returns:
        None
    """

    cache_file = Path(processed_path) / "cached_queries.pkl"

    try:
        with open(cache_file, "rb") as f:
            cached_queries = pickle.load(f)
    except FileNotFoundError:
        cached_queries = {}
    except Exception as e:
        print_red(f"Error: during loading cached queries - {e}")
        sys.exit(1)

    cached_queries[query] = result

    with open(cache_file, "wb") as f:
        pickle.dump(cached_queries, f)


def check_if_query_cached(
    query: str, processed_path: str
) -> list[Chunk] | None:
    """
    Checks if a query is already cached and returns the result if it is.

    Args:
        query (str): The query string to check.
        processed_path (str): The path to the processed data.

    Returns:
        list[Chunk] | None: The cached result if it exists, None otherwise.
    """

    cache_file = Path(processed_path) / "cached_queries.pkl"

    try:
        with open(cache_file, "rb") as f:
            cached_queries: dict[str, list] = pickle.load(f)

        if query in cached_queries:
            return cached_queries[query]
        return None

    except FileNotFoundError:
        return None
    except Exception as e:
        print_red(f"Error: during loading cached queries - {e}")
        sys.exit(1)
