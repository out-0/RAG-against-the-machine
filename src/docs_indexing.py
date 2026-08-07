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
from pathlib import Path
import threading
import bm25s
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from src.custom_print import print_green, print_yellow, print_red
from src.data_models import Chunk
from src.vector_idx import v_idx_build_and_save


def indexing(
    chunks: list[Chunk] | None = None,
    processed_path: str = "data/processed",
    # method: str = "bm25",
    use_embedding: bool = False,
    embeddings_model_name: str | None = "all-MiniLM-L6-v2",
    use_hybrid: bool = False,
) -> None:
    """Run the indexer based on the indexing method specified by args
    if not specified, the default is bm25 which is implemented by
    bm25s library

    if hypbrid we just run both (keyword | embeddings) indexing
    """

    if not chunks:
        raise TypeError("Warning: Chunks to index is missed")

    # Create the path for the indexed lookup
    Path(processed_path).mkdir(parents=True, exist_ok=True)

    # Extract content from chunks instead of full obj
    docs: list[str] = [chunk.content for chunk in chunks]

    # Splite keyword indexing and embeddings to use them as threads

    def keyword_indexing(docs: list[str]) -> None:
        """"""
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
        docs: list[str],
    ) -> None:
        """"""
        if embeddings_model_name != "all-MiniLM-L6-v2":
            print_yellow(
                "WARNING: Currently only 'all-MiniLM-L6-v2' is supported for embedding"
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
            args=(embeddings_model_name, processed_path, docs),
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
                docs=docs,
            )

        # Should not triggered
        else:
            print_red("Error: Somehow something happen")
            return

    # Save chunks as pickle file (binary format) so can
    # be loaded also later to map result to chunk obj
    with open(Path(processed_path) / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
        print("Chunks saved as pickle file for later mapping to results")


# class Tf_idf_search:
#     """Term Frequency-Inverse Document Frequency
#         A simple algo that measures how often a word appears in a document.
#         A higher frequency suggests greater importance.
#         If a term appears frequently in a document, it is likely relevant
#         to the document’s content.
#
#     Formula is:
#         TF(t,d) =   number of times term appears in document /
#                     total number of words in document
#
#         IDF(t,D) = log(Total documents count / Documents have the term)
#
#     The more a word appears in a single document,
#     the more important that word is for that document,
#     the more it appears in the corpus then the less
#     important that word is overall.
#     """
#
#     def __init__(self, chunks: list[Chunk]) -> None:
#         """"""
#         self.chunks: list[Chunk] = chunks
#
#         documents: list[str] = [chunk.content for chunk in self.chunks]
#
#         self.vectorizer = TfidfVectorizer()
#         self.tf_idf_matrix = self.vectorizer.fit_transform(self.documents)
#
#         self.cleaner_docs: dict[Chunk, list[str]] = self._cleaning_chunks(docs=chunks)
#
#     def run(self, query: str) -> None:
#         """"""
#         pass
#
#     def _cleaning_chunks(self, docs: list[Chunk]) -> dict[Chunk, list[str]]:
#         """
#         Cleaning the docs by removing punctuations and tokenize them into words
#         for simpler processing later,
#
#         Args:
#             - docs: list of documents you'll process
#         Returns:
#             - list of the tokenized words for each doc
#         """
#         cleaner_map: dict[Chunk, list[str]] = {}
#
#         import re
#
#         for doc in docs:
#             tokens: list[str]
#
#             normalized = re.sub(r"[^\w\s]", "", doc.content)
#             tokens = normalized.lower().split()
#
#             cleaner_map[doc] = tokens
#
#         return cleaner_map
#
#     def term_frequency(self, term: str, document_words: list[str]) -> float:
#         """"""
#         tokens: list[str] = document_words
#         if not tokens:
#             return 0.0
#         return tokens.count(term) / len(tokens)
#
#     def inverse_document_frequency(self, term: str) -> float:
#         """"""
#         count_of_documents: float = len(self.cleaner_docs) + 1
#         count_of_documents_with_term: float = (
#             sum([1 for doc in self.cleaner_docs if term in self.cleaner_docs[doc]]) + 1
#         )
#         idf: float = math.log10(count_of_documents / count_of_documents_with_term) + 1
#         return idf
#
#     def score_document(self, query: str, document_words: list[str]) -> float:
#         """"""
#         quwery_words: list[str] = query.lower().split()
#         score: float = 0.0
#
#         for term in quwery_words:
#             tf: float = self.term_frequency(term, document_words)
#             idf: float = self.inverse_document_frequency(term=term)
#
#             score += tf * idf
#
#         return score
#
#     def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
#         """Search for a query again the docs available and return
#         top k docs after sorting them based on the score gained
#         from the tf-idf process
#
#         Args:
#             - query: user question
#             - tok_k: limit of docs you want to return
#
#         Returns:
#             list of top k documents
#         """
#         results: list[tuple[Chunk, float]] = []
#
#         for doc in self.cleaner_docs:
#             score: float = self.score_document(query, self.cleaner_docs[doc])
#             if score > 0:
#                 results.append((doc, score))
#
#         # Sort by score in descending order (highest score first)
#         results.sort(key=lambda x: x[1], reverse=True)
#         if top_k > len(results):
#             raise ValueError(
#                 "Warrning: Not enough docs to be returned", "try smaller top_k"
#             )
#         return results[:top_k]
