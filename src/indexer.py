import math
from docs_chunking import Chunk
from collections import Counter


class Indexer:
    """"""

    def tf_idf(self) -> None:
        pass


class Tf_idf:
    """Term Frequency-Inverse Document Frequency
        A simple algo that measures how often a word appears in a document.
        A higher frequency suggests greater importance.
        If a term appears frequently in a document, it is likely relevant
        to the document’s content.

    Formula is:
        TF(t,d) =   number of times term appears in document /
                    total numer of words in document

        IDF(t,D) = log(Total documents count / Documents have the term)
    """

    def run(self) -> None:
        """"""
        pass

    def remove_punctuations(self, chunk: Chunk) -> Chunk:
        """iterate over the doc content and remove the punctuations"""
        punctuations = ".,;:!?"

        for punc in punctuations:
            chunk.content = chunk.content.replace(punc, "")

        return chunk

    def tokenize(self, chunk: Chunk) -> list[str]:
        """Split doc into tokens(words)"""
        return chunk.content.lower().split()

    def build_tf(self, chunks: list[Chunk]) -> dict[int, dict[str, int]]:
        """Build document frequency and term frequency"""

        # Term Frequency which store how much a word appear in document(chunk)
        docs_terms_frequency: dict[int, dict[str, int]] = {}

        # For each chunk register the frequency of each word
        for chunk in chunks:
            docs_terms_frequency[chunk.id] = Counter(self.tokenize(chunk))

        return docs_terms_frequency
