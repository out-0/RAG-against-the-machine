class Indexer:
    """"""

    def tf_idf(self) -> None:
        """Term Frequency-Inverse Document Frequency
         A simple algo that measures how often a word appears in a document.
         A higher frequency suggests greater importance.
         If a term appears frequently in a document, it is likely relevant
         to the document’s content.

        Formula is:
            TF(t,d) =   number of times term appears in document /
                        total terms number in document

            IDF(t,D) = log(Total documents count / Documents have the term)
        """
