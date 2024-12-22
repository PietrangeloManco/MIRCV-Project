import unittest
import math
from Lexicon.Lexicon import Lexicon
from DocumentTable.DocumentTable import DocumentTable
from Query.Scoring import Scoring


class TestScoring(unittest.TestCase):
    def setUp(self):
        # Set up actual Lexicon and DocumentTable objects with real data
        self.lexicon = Lexicon()
        self.document_table = DocumentTable()

        # Add terms and documents to the Lexicon and DocumentTable
        self.lexicon.add_term("term1", 10)
        self.lexicon.add_term("term2", 15)

        self.document_table.add_document(1, 100)
        self.document_table.add_document(2, 200)
        self.document_table.add_document(3, 150)

        # Instantiate Scoring
        self.scoring = Scoring(self.lexicon, self.document_table)

    def test_calculate_avg_doc_length(self):
        """Test the calculation of the average document length."""
        expected_avg_doc_length = (100 + 200 + 150) / 3
        self.assertAlmostEqual(self.scoring.avg_doc_length, expected_avg_doc_length)

    def test_compute_tfidf(self):
        """Test the computation of TF-IDF score."""
        term = "term1"
        payload = 2  # Example term frequency
        # Expected IDF calculation (using provided formula)
        idf = math.log(self.scoring.total_documents / self.lexicon.get_term_info(term))
        expected_tfidf = (1 + math.log(payload)) * idf
        tfidf_score = self.scoring.compute_tfidf(term, payload)
        self.assertAlmostEqual(tfidf_score, expected_tfidf)

    def test_compute_bm25(self):
        """Test the computation of BM25 score."""
        term = "term1"
        doc_id = 1
        payload = 2  # Example term frequency
        k1 = 1.5
        b = 0.75

        doc_length = self.document_table.get_document_length(doc_id)
        idf = math.log(self.scoring.total_documents / self.lexicon.get_term_info(term))
        numerator = payload
        denominator = payload + k1 * (1 - b + b * (doc_length / self.scoring.avg_doc_length))
        expected_bm25 = idf * (numerator / denominator)

        bm25_score = self.scoring.compute_bm25(term, doc_id, payload, k1, b)
        self.assertAlmostEqual(bm25_score, expected_bm25)

    def test_compute_score_tfidf(self):
        """Test the compute_score method with tfidf scoring."""
        term = "term1"
        doc_id = 1
        payload = 2
        score = self.scoring.compute_score(term, doc_id, payload, method="tfidf")

        expected_score = self.scoring.compute_tfidf(term, payload)
        self.assertAlmostEqual(score, expected_score)

    def test_compute_score_bm25(self):
        """Test the compute_score method with bm25 scoring."""
        term = "term1"
        doc_id = 1
        payload = 2
        score = self.scoring.compute_score(term, doc_id, payload, method="bm25")

        expected_score = self.scoring.compute_bm25(term, doc_id, payload)
        self.assertAlmostEqual(score, expected_score)


if __name__ == "__main__":
    unittest.main()
