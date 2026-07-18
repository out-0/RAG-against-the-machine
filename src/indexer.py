import math
from src.docs_chunking import Chunk
from collections import Counter
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bm25s
from pathlib import Path
import tqdm


class Indexer:
    """Index class wrapper for handling choosed chunking method"""

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        method: str = "bm25",
        processed_path: str = "data/processed",
    ) -> None:
        """Setup indexer to index the chunks of docs and build
        the lookup table, which can be used later again the query
        to retrieve the top k docs that relavant to that query

        Args:


        Returns:

        """

        if not chunks:
            raise TypeError("Warrning: Chunks to index is missed")

        self.chunks: list[Chunk] = chunks
        self.processed_path: str = processed_path
        self.method: str = method

        # Create the path for the indexed lookup
        Path(self.processed_path).mkdir(parents=True, exist_ok=True)

        # Extract content from chunks instead of full obj
        docs: list[str] = [chunk.content for chunk in self.chunks]

        if method == "bm25":
            # A stimmer ot normalize words
            # I dont think i need it since we are processing python
            # and markdown files so normalize generaling thw words
            # if making it even better to match the text code.
            #
            # stemmer = Stemmer.stemmer("english")

            # Tokenize the corpus and only keep the ids
            # (faster and saves memory)
            # It may return tokenized obj which hold (ids, vocab)
            # ids for docs and vocab map each word to id
            corpus_tokens = bm25s.tokenize(texts=docs)

            # print(docs)
            # print(type(docs))
            # print(id_vocab)
            # print(type(id_vocab))

            # Register the chunks so they got returned later after reteriving
            retriever: bm25s.BM25 = bm25s.BM25(corpus=list(range(len(self.chunks))))

            # Index the docs
            retriever.index(corpus=corpus_tokens, show_progress=True)

            print(type(retriever.vocab_dict))

            k = next(iter(retriever.vocab_dict))
            print(type(k), repr(k))

            # Store it to use it later for retreiving
            self.bm_retriever = retriever

            # Save the indexing for fast reterival later
            retriever.save(self.processed_path)
            exit()

        elif method == "tf_idf":
            pass

    def bm25_search(self, query: str) -> None:
        """"""

        pass


class TfidfSearch:
    """Retrieves relevant document chunks using TF-IDF and Cosine Similarity.

    This implementation leverages scikit-learn to handle the TF-IDF calculation
    as a sparse matrix under the hood. This makes it highly optimized and fast
    for large corpora (e.g., 20,000+ chunks) compared to pure Python loops.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        """Initializes the TF-IDF matrix based on the provided text chunks.

        Args:
            chunks: A list of Chunk objects containing the text to be indexed.
        """
        if not chunks:
            self.chunks: list[Chunk] = []
            self.vectorizer: TfidfVectorizer = TfidfVectorizer()
            self.tfidf_matrix: Any = self.vectorizer.fit_transform(raw_documents=[""])
            return

        self.chunks = chunks

        # Extract raw text content from the custom Chunk objects
        documents: list[str] = [chunk.content for chunk in self.chunks]

        # Initialize the Vectorizer
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,  # Applies 1 + log(TF) to cap overly frequent terms
            token_pattern=r"(?u)\b\w\w+\b",  # Standard pattern to grab words with 2+ alphanumeric chars
        )

        # Fit the vectorizer to the documents and create the TF-IDF sparse matrix.
        self.tfidf_matrix = self.vectorizer.fit_transform(raw_documents=documents)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Searches the indexed chunks for the most relevant matches to the query.

        Args:
            query: The search string entered by the user.
            top_k: The maximum number of top-scoring chunks to return.

        Returns:
            A list of tuples, where each tuple contains:
                - The original Chunk object.
                - Its corresponding cosine similarity score (float).
            Sorted in descending order of score.
        """
        if not self.chunks:
            return []

        # Transform the user query into a TF-IDF vector using the LEARNED vocabulary.
        # We must use the exact same vocabulary and IDF weights learned during initialization.
        query_vector: Any = self.vectorizer.transform([query])

        # Calculate cosine similarity between the query vector and all document vectors.
        # Returns a 2D array like [[0.12, 0.00, 0.85, 0.04, ...]]
        similarity_scores: Any = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Pair the calculated scores with their corresponding original Chunk objects
        results: list[tuple[Chunk, float]] = []
        for index, score in enumerate(similarity_scores):
            # Only include chunks that have at least some mathematical overlap
            if score > 0.0:
                results.append((self.chunks[index], float(score)))

        # Sort the results by score in descending order (highest score first)
        results.sort(key=lambda x: x[1], reverse=True)

        # Safely slice the top_k results. If fewer results matched than top_k,
        # it just returns however many it found without throwing a ValueError.
        return results[:top_k]


# class Tf_idf_search:
#     """Term Frequency-Inverse Document Frequency
#         A simple algo that measures how often a word appears in a document.
#         A higher frequency suggests greater importance.
#         If a term appears frequently in a document, it is likely relevant
#         to the document’s content.
#
#     Formula is:
#         TF(t,d) =   number of times term appears in document /
#                     total numer of words in document
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
