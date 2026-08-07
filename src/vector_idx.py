from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.custom_print import print_green, print_yellow
from src.docs_chunking import Chunk

# FAISS search class for semantic search using embeddings
# Also store the vectors in FAISS index for efficient retrieval


def v_idx_build_and_save(
    model_name: str, index_save_dir: str, docs: list[str]
) -> None:
    """Encodes docs, builds the FAISS index, and saves it."""

    # Load the embedding model
    model = SentenceTransformer(model_name)

    docs: list[str] = []

    # count the number of tokens in each chunk to see if any exceed the model's max sequence length
    reducing_needed: int = 0
    token_lengths: list[int] = []
    for doc in docs:
        chunk_token_len: int = len(model.tokenizer.encode(doc, verbose=False))
        token_lengths.append(chunk_token_len)
        docs.append(doc)
        if chunk_token_len > model.max_seq_length:
            reducing_needed = 1

    print("\n===== State =====")
    print(f"Embedding model : {model_name}")
    print(f"Max sequence    : {model.max_seq_length} tokens")

    print("Dataset:")
    print(f"    Average: {sum(token_lengths) / len(token_lengths):.1f}")
    print(f"    Max: {max(token_lengths)}")
    print(f"    Min: {min(token_lengths)}")
    print()
    if reducing_needed:
        print_yellow(
            f"WARNING: Some chunks exceed the model's max sequence length of {model.max_seq_length} tokens."
        )
        print_green("Recommendation:")
        print("     Reduce --max_chunk_size to around 900-1100 characters.")
        print(
            "     Otherwise, some chunks will be truncated and may lose some information."
        )
    print("===== ===== =====\n")

    # Embeed documents
    embeddings = model.encode_document(
        docs,
        show_progress_bar=True,
    )

    # Create FAISS index table.
    index = faiss.IndexFlatIP(embeddings.shape[0])

    # Add the vectors to the index
    index.add(embeddings)

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
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
    chunks: list[Chunk],
) -> list[tuple[Chunk, float]]:
    """Searches the FAISS index. Extremely clean."""
    if index is None:
        raise ValueError("Index not loaded!")

    # Turn query into a 2d numpy array with shape [num_inputs, output_dimension]
    query_vector = model.encode(
        [query],
    )

    # Search (Returns distances and IDs)
    distances, indices = index.search(query_vector, k)
    print(distances)
    print(indices)
    import sys
    sys.exit()

    # Format results
    results = [
        chunks[idx] for idx in indices[0] if idx != -1
    ]  # Filter out invalid indices
    # distances and indices are 2D arrays like [[0.1, 0.5, 0.8]], so we take [0]

    return results
