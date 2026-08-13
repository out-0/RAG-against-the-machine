import hashlib
import pickle
from pathlib import Path

import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_models import Chunk
from src.docs_chunking import Chunker
from src.docs_indexing import indexing
from src.vector_idx import v_idx_load


def handle_incremental_indexing(
    docs: list,
    processed_path: str = "data/processed/",
    max_chunk_size: int = 2000,
    use_embedding: bool = False,
    embeddings_model_name: str | None = "all-MiniLM-L6-v2",
) -> None:
    """Perform incremental indexing based on per-file content hashes.

    Changed or newly added files are re-chunked and re-embedded.
    Deleted files are removed from the FAISS and chunk store.
    BM25 is rebuilt from the updated chunk set on every incremental run.
    """
    processed_dir = Path(processed_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    hash_path: Path = processed_dir / "files_hash.pkl"

    if not hash_path.exists():
        print("No previous hash metadata found. Performing full indexing.")
        full_index(
            docs,
            processed_path=processed_path,
            max_chunk_size=max_chunk_size,
            use_embedding=use_embedding,
            embeddings_model_name=embeddings_model_name,
        )
        return

    # Load the previous hash metadata
    with open(hash_path, "rb") as f:
        old_metadata: dict[str, str] = pickle.load(f)

    changed_files: list = []
    current_paths = set()
    # Check for changed or newly added files
    for doc in docs:
        current_paths.add(doc.path)
        new_hash = generate_hash(doc.content)
        if old_metadata.get(doc.path) != new_hash:
            changed_files.append(doc)

    # Check for deleted files
    deleted_paths = set(old_metadata.keys()) - current_paths

    # No changes detected
    if not changed_files and not deleted_paths:
        print("No changes detected.")
        return

    chunks_path = processed_dir / "chunks.pkl"
    index_path = processed_dir / "index.faiss"
    if not chunks_path.exists():
        print(
            "Existing chunk metadata is missing. "
            "Falling back to full indexing."
        )
        full_index(
            docs,
            processed_path=processed_path,
            max_chunk_size=max_chunk_size,
            use_embedding=use_embedding,
            embeddings_model_name=embeddings_model_name,
        )
        return

    with open(chunks_path, "rb") as f:
        chunks: list[Chunk] = pickle.load(f)

    # Load the FAISS index with safety fallback if somehow index is missing
    if use_embedding:
        if not index_path.exists():
            print(
                "Existing FAISS index is missing. "
                "Falling back to full indexing."
            )
            full_index(
                docs,
                processed_path=processed_path,
                max_chunk_size=max_chunk_size,
                use_embedding=use_embedding,
                embeddings_model_name=embeddings_model_name,
            )
            return
        index = v_idx_load(str(index_path))

    # Remove affected chunks for changed/deleted paths.
    removed_chunk_ids: set[int] = set()
    # Merge the two sets eliminating any duplicates
    affected_paths = {doc.path for doc in changed_files} | deleted_paths

    # Collect removed affected chunks
    for chunk in chunks:
        if chunk.file_path in affected_paths:
            removed_chunk_ids.add(chunk.id)

    # Remove affected chunks from vector index
    if removed_chunk_ids:
        if use_embedding:
            selector = faiss.IDSelectorBatch(
                np.array(list(removed_chunk_ids), dtype=np.int64)
            )
            index.remove_ids(selector)
        # Collect not removed chunks to be used below
        chunks = [
            chunk for chunk in chunks if chunk.id not in removed_chunk_ids
        ]

    # Process changed files only and re-add their embedding vectors.
    if changed_files:
        # Get the last know idx so new chunks can be assigned next unique ids
        next_id = max((chunk.id for chunk in chunks), default=0) + 1

        # Initialize embedding model
        if use_embedding:
            embedding_model = SentenceTransformer(
                embeddings_model_name or "all-MiniLM-L6-v2"
            )

        # Chunk changed files
        doc_chunker = Chunker(files=changed_files, max_size=max_chunk_size)
        new_chunks = doc_chunker.process_files()
        for chunk in new_chunks:
            chunk.id = next_id
            next_id += 1

        # Embed new chunks and add them to the vector index
        if new_chunks:
            if use_embedding:
                new_embeddings = embedding_model.encode(
                    [chunk.content for chunk in new_chunks],
                    show_progress_bar=True,
                    convert_to_numpy=True,
                ).astype("float32")
                new_ids = np.array(
                    [chunk.id for chunk in new_chunks], dtype=np.int64
                )
                # Add new embeddings to vector index in memory
                index.add_with_ids(new_embeddings, new_ids)
            # Append new chunks so all chunks can be used below for BM25
            chunks.extend(new_chunks)

    # Rebuild BM25 from scratch using the updated chunks list.
    rerun_bm25_indexing(chunks=chunks, processed_path=processed_path)

    # Persist updated state.
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    # Persist updated FAISS index in disk
    if use_embedding and index is not None:
        faiss.write_index(index, str(index_path))
    hash_documents(docs, processed_path)
    print("Incremental indexing complete.")


def full_index(
    docs: list,
    processed_path: str,
    max_chunk_size: int,
    use_embedding: bool,
    embeddings_model_name: str | None,
) -> None:
    chunker = Chunker(files=docs, max_size=max_chunk_size)
    chunks: list[Chunk] = chunker.process_files()
    indexing(
        chunks=chunks,
        processed_path=processed_path,
        use_embedding=use_embedding,
        embeddings_model_name=embeddings_model_name,
    )
    hash_documents(docs, processed_path)


def rerun_bm25_indexing(chunks: list[Chunk], processed_path: str) -> None:
    corpus_tokens = bm25s.tokenize(
        texts=[chunk.content for chunk in chunks], show_progress=True
    )
    retriever = bm25s.BM25(k1=1.5, b=0.5)
    retriever.index(corpus=corpus_tokens, show_progress=True)
    retriever.save(processed_path)


# To hash documents content
def generate_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def hash_documents(docs: list, processed_path: str) -> None:
    """
    loop on passed documents and generate hash for each file content
    """
    metadata: dict[str, str] = {}
    for doc in docs:
        metadata[doc.path] = generate_hash(doc.content)

    with open(Path(processed_path) / "files_hash.pkl", "wb") as f:
        pickle.dump(metadata, f)
