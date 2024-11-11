import math
from Lexicon.Lexicon import Lexicon
from DocumentTable.DocumentTable import DocumentTable

class Scoring:
    def __init__(self, lexicon: Lexicon, document_table: DocumentTable):
        self.lexicon = lexicon
        self.document_table = document_table
        self.total_documents = len(document_table.get_all_documents())
        self.avg_doc_length = self._calculate_avg_doc_length()

    def _calculate_avg_doc_length(self) -> float:
        """Calculates the average document length for BM25 normalization."""
        total_length = sum(
            self.document_table.get_document_length(doc_id) for doc_id in self.document_table.get_all_documents())
        return total_length / self.total_documents if self.total_documents > 0 else 0

    def compute_tfidf(self, term: str, doc_id: int) -> float:
        """Computes the TFIDF score for a given term in a document."""
        term_info = self.lexicon.get_term_info(term)
        if not term_info:
            return 0.0  # Term not found in the lexicon

        # Retrieve term frequency and document length
        term_frequency = term_info['term_frequency']
        doc_length = self.document_table.get_document_length(doc_id)
        if doc_length == 0:
            return 0.0  # Avoid division by zero

        # Compute normalized term frequency (TF)
        tf = term_frequency / doc_length

        # Compute inverse document frequency (IDF)
        idf = math.log((self.total_documents + 1) / (term_info['term_frequency'] + 1)) + 1  # Smoothing with +1

        # TFIDF score is TF * IDF
        return tf * idf

    def compute_bm25(self, term: str, doc_id: int, k1: float = 1.5, b: float = 0.75) -> float:
        """Computes the BM25 score for a given term in a document."""
        term_info = self.lexicon.get_term_info(term)
        if not term_info:
            return 0.0  # Term not found in the lexicon

        # Retrieve document length
        doc_length = self.document_table.get_document_length(doc_id)
        if doc_length == 0:
            return 0.0  # Avoid division by zero

        # Compute normalized term frequency (TF)
        term_frequency = term_info['term_frequency']
        tf = term_frequency / doc_length

        # BM25-specific IDF
        idf = math.log(
            (self.total_documents + 1) / (term_info['term_frequency'] + 1)) + 1  # Smoothing with +1

        # BM25 formula: idf * (numerator / denominator)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
        bm25_score = idf * (numerator / denominator)

        return bm25_score

    def compute_score(self, term: str, doc_id: int, method: str = "tfidf") -> float:
        """Computes the score for a term in a document using the specified method."""
        if method == "tfidf":
            return self.compute_tfidf(term, doc_id)
        elif method == "bm25":
            return self.compute_bm25(term, doc_id)
        else:
            raise ValueError("Invalid scoring method. Choose 'tfidf' or 'bm25'")
