from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.custom_print import print_green, print_yellow
from src.docs_chunking import Chunk

# FAISS search is used for semantic search using embeddings
# Also store the vectors in FAISS index for efficient retrieval


def v_idx_build_and_save(
    model_name: str, index_save_dir: str, chunks: list[Chunk]
) -> None:
    """Encodes docs, builds the FAISS index, and saves it.

    Args:
        model_name: sentence-transformers model name
        index_save_dir: directory to save index.faiss
        chunks: list of Chunk objects to embed and index
    """

    # Load the embedding model
    model = SentenceTransformer(model_name)

    if not chunks:
        raise ValueError("No documents provided to build the index")

    docs = [chunk.content for chunk in chunks]
    ids = np.array([chunk.id for chunk in chunks], dtype=np.int64)

    # count the number of tokens in each chunk to see if any exceed the model's
    # max sequence length
    reducing_needed: int = 0
    token_lengths: list[int] = []

    # A Useless loop to help for printing the formated state below
    for doc in docs:
        try:
            chunk_token_len: int = len(
                model.tokenizer.encode(doc, verbose=False)
            )

        except Exception:
            chunk_token_len = len(doc.split())

        token_lengths.append(chunk_token_len)
        if chunk_token_len > getattr(model, "max_seq_length", 512):
            reducing_needed = 1

    print("\n===== State =====")
    print(f"Embedding model : {model_name}")
    print(
        f"Max sequence: {getattr(model, 'max_seq_length', 'unknown')} tokens"
    )
    print("Dataset:")
    print(f"    Average: {sum(token_lengths) / len(token_lengths):.1f}")
    print(f"    Max: {max(token_lengths)}")
    print(f"    Min: {min(token_lengths)}")
    print()
    if reducing_needed:
        print_yellow(
            f"WARNING: Some chunks exceed the model's max sequence length "
            f"of {getattr(model, 'max_seq_length', 'unknown')} tokens."
        )
        print_green("Recommendation:")
        print("  Reduce --max_chunk_size to around 900-1100 characters.")
        print(
            "   Otherwise, some chunks will be truncated and may lose "
            "   some information."
        )
    print("===== ===== =====\n")

    # Embed documents
    embeddings = model.encode(
        docs,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Ensure float32
    embeddings = embeddings.astype("float32")

    # Create FAISS index table with correct dimension and stable ids
    # dim is vector dimension size (the number of float numbers that
    # make up a single vector)
    dim = embeddings.shape[1]
    # The base engine that calculates similarity
    base_index = faiss.IndexFlatIP(dim)
    # The index table to map vectors to chunk ids (for incremental updates)
    index = faiss.IndexIDMap2(base_index)

    # Add the vectors to the index with stable chunk ids
    index.add_with_ids(embeddings, ids)

    # Save to a single file
    faiss.write_index(index, str(Path(index_save_dir) / "index.faiss"))
    print_green(f"Saved FAISS index to {index_save_dir}")


def v_idx_load(index_path: str) -> faiss.Index:
    """Loads the FAISS index from disk."""
    index: faiss.Index = faiss.read_index(index_path)
    print(f"Loaded FAISS index from {index_path}")
    return index


def v_idx_search(
    query: str,
    k: int,
    index: faiss.Index,
    model: SentenceTransformer,
    chunks: list[Chunk],
) -> list[tuple[Chunk, float]]:
    """Searches the FAISS index and returns list of (Chunk, score).

    Scores are the raw distances returned
    by FAISS (inner product similarity for IndexFlatIP).
    """
    # Turn query into a 2d numpy
    # array with shape [num_inputs, output_dimension]
    query_vector = model.encode([query], convert_to_numpy=True)
    query_vector = query_vector.astype("float32")

    # Search (Returns distances and IDs)
    score_distances, chunk_ids = index.search(query_vector, k)

    # A lookup table map chunk ids to chunks to extract the real chunks from
    # the ids returned from faiss
    chunk_lookup = {chunk.id: chunk for chunk in chunks}
    results: list[tuple[Chunk, float]] = []

    for dist, idx in zip(score_distances[0], chunk_ids[0]):
        if idx == -1:
            continue
        score = float(dist)
        # Just a defensive check
        if idx not in chunk_lookup:
            continue
        results.append((chunk_lookup[idx], score))

    return results
