import math

from DocumentTable.DocumentTable import DocumentTable
from Lexicon.Lexicon import Lexicon


class Scoring:
    def __init__(self, lexicon: Lexicon, document_table: DocumentTable):
        """
        Initialization for the scoring class.

        Args:
            lexicon: Lexicon instance.
            document_table: Document Table instance.
        """
        self.lexicon = lexicon
        self.document_table = document_table
        self.total_documents = len(document_table.get_all_documents())
        self.avg_doc_length = self._calculate_avg_doc_length()

    def _calculate_avg_doc_length(self) -> float:
        """
        Calculates the average document length for BM25 normalization.

        Returns:
            float: Average collection document length.
        """
        total_length = sum(
            self.document_table.get_document_length(doc_id) for doc_id in self.document_table.get_all_documents())
        return total_length / self.total_documents if self.total_documents > 0 else 0

    def compute_tfidf(self, term: str, payload: int) -> float:
        """
        Computes the TFIDF score for a given term in a document.

        Args:
            term(str): The query term.
            payload(int): The frequency of the term in the document.

        Returns:
            float: The TFIDF of the term, document pair.
        """

        term_frequency = payload

        tf = 1 + math.log(term_frequency)

        # Compute inverse document frequency (IDF)
        idf = math.log(self.total_documents / (self.lexicon.get_term_info(term)))

        # TFIDF score is TF * IDF
        return tf * idf

    def compute_bm25(self, term: str, doc_id: int, payload: int, k1: float = 1.5, b: float = 0.75) -> float:
        """
        Computes the BM25 score for a given term in a document.

        Args:
            term(str): The query term.
            doc_id(int): The document ID.
            payload(int): The frequency of the term in the document.
            k1(float): BM25 parameter. Default 1.5 (usually in the range 1.2-2.0)
            b(float): BM25 length normalization parameter. Default 0.75 (0 is no length normalization,
            1 is full length normalization).

        Returns:
            float: The BM25 score of the term, document pair.
        """

        # Retrieve document length
        doc_length = self.document_table.get_document_length(doc_id)

        if doc_length == 0:
            return 0.0  # Avoid division by zero

        tf = payload

        idf = math.log(
            self.total_documents / (self.lexicon.get_term_info(term)))

        # BM25 formula: idf * (numerator / denominator)
        numerator = tf
        denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
        bm25_score = idf * (numerator / denominator)

        return bm25_score

    def compute_score(self, term: str, doc_id: int, payload: int, method: str = "tfidf") -> float:
        """
        Computes the score by calling the proper ranking method.

        Args:
            term(str): Query term.
            doc_id(int): Document id.
            payload(int): Term frequency in the given document.
            method(str): TFIDF of BM25. Default "tfidf".

        Returns:
            float: Either TFIDF of BM25 score, depending on the chosen method.
        """
        if method == "tfidf":
            score = self.compute_tfidf(term, payload)
        elif method == "bm25":
            score = self.compute_bm25(term, doc_id, payload)
        else:
            raise ValueError("Invalid scoring method. Choose 'tfidf' or 'bm25'")
        return score
