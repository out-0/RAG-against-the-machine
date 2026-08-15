"""Index class wrapper for handling chosen chunking method,

indexing is basically create a special map that link each
word to the relevant docs for it after scoring it by a
some formula's based on the indexing method,

To simple it, Index the docs internally build a map 'the inverted index'
which map a word to list of matching documents and the frequency
of that word withing that document, something like:
"vllm"      -----> [ Doc 0, Doc 2 ]
"paged"     -----> [ Doc 0, Doc 1 ]
"uses"      -----> [ Doc 0 ]
"attention" -----> [ Doc 0, Doc 1 ]

Its more than that but,
also that is just implied for a lexical indexing like TF-IDF, BM25,
But for an embedding its relay on a vectors which kinda based on
the attentions between the terms or phrases
"""

import pickle
import threading
from pathlib import Path

import bm25s
from huggingface_hub import logging

from src.custom_print import print_green, print_red, print_yellow
from src.data_models import Chunk
from src.vector_idx import v_idx_build_and_save

# Turn off the huggingface unauthenticated warning
logging.set_verbosity_error()


def indexing(
    chunks: list[Chunk] | None = None,
    processed_path: str = "data/processed",
    use_embedding: bool = False,
    embeddings_model_name: str | None = "all-MiniLM-L6-v2",
    use_hybrid: bool = False,
) -> None:
    """
    Indexing which create a special map that link each
    word to the relevant docs for it after scoring it by a
    some formula's based on the indexing method,

    Args:
        - chunks (list[Chunk]): The list of chunks
        - processed_path (str): The path to save the indexed lookup
        - use_embedding (bool): Whether to use embeddings
        - embeddings_model_name (str): The name of the embeddings model
        - use_hybrid (bool): Whether to use hybrid indexing

    Returns:
        - None
    """

    if not chunks:
        raise TypeError("Warning: Chunks to index is missed")

    # Create the path for the indexed lookup
    Path(processed_path).mkdir(parents=True, exist_ok=True)

    # Extract content from chunks instead of full obj
    docs: list[str] = [chunk.content for chunk in chunks]

    # Splite keyword indexing and embeddings to use them as threads

    def keyword_indexing(docs: list[str]) -> None:
        """
        Create a special map that link each word to the relevant docs for it
        by bm25
        """
        corpus_tokens: list[list[str]] | bm25s.tokenization.Tokenized = (
            bm25s.tokenize(texts=docs, show_progress=True)
        )

        # b: for length penalties(long files got penalities)
        # k1: word repetition boosting
        retriever: bm25s.BM25 = bm25s.BM25(k1=1.5, b=0.5)
        retriever.index(corpus=corpus_tokens, show_progress=True)
        retriever.save(processed_path)

        print_green(
            f"Ingestion complete! "
            f"Indexed {len(chunks)} chunks under {processed_path}"
        )

    def semantic_indexing(
        embeddings_model_name: str,
        processed_path: str,
    ) -> None:
        """
        Create a special map that link each word to the relevant docs for it
        by embeddings
        """
        if embeddings_model_name != "all-MiniLM-L6-v2":
            print_yellow(
                "WARNING: Currently only 'all-MiniLM-L6-v2' "
                "is supported for embedding"
            )
            print_yellow("Fallback to default 'all-MiniLM-L6-v2'")
            embeddings_model_name = "all-MiniLM-L6-v2"

        # Build and save the FAISS index for semantic search
        v_idx_build_and_save(
            model_name=embeddings_model_name,
            index_save_dir=processed_path,
            chunks=chunks,
        )

    if use_hybrid:
        t1 = threading.Thread(target=keyword_indexing, args=(docs,))
        t2 = threading.Thread(
            target=semantic_indexing,
            args=(embeddings_model_name, processed_path),
        )
        t1.start()
        t2.start()
        # Wait for them to finish
        t1.join()
        t2.join()
    else:
        # This use bm25
        if not use_embedding:
            keyword_indexing(docs=docs)

        # This Embedding model
        elif use_embedding and embeddings_model_name:
            semantic_indexing(
                embeddings_model_name=embeddings_model_name,
                processed_path=processed_path,
            )

        # Should not triggered
        else:
            print_red("Error: Somehow something happen\n")
            return

    # Save chunks as pickle file (binary format) so can
    # be loaded also later to map result to chunk obj
    with open(Path(processed_path) / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
        # print("Chunks saved as pickle file for later mapping to results")
