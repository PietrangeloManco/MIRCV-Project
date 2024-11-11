import unittest
from Lexicon.Lexicon import Lexicon
from DocumentTable.DocumentTable import DocumentTable
from Query import Scoring  # Assuming Scoring class is saved as Scoring.py

class TestScoring(unittest.TestCase):
    def setUp(self):
        """Set up the test environment with a lexicon and a document table."""
        # Initialize the Lexicon and DocumentTable
        self.lexicon = Lexicon()
        self.document_table = DocumentTable()

        # Add sample terms to the Lexicon
        self.lexicon.add_term("example", 0, 5)  # Term "example" with 5 occurrences
        self.lexicon.add_term("test", 10, 8)    # Term "test" with 8 occurrences
        self.lexicon.add_term("search", 20, 3)  # Term "search" with 3 occurrences

        # Add sample documents to the DocumentTable
        self.document_table.add_document(1, 100, 0)  # Document 1 with 100 terms
        self.document_table.add_document(2, 150, 200)  # Document 2 with 150 terms
        self.document_table.add_document(3, 50, 400)   # Document 3 with 50 terms

        # Initialize the Scoring class
        self.scoring = Scoring.Scoring(self.lexicon, self.document_table)


    def test_tfidf_scoring(self):
        """Test TFIDF scoring for various terms and documents."""
        score = self.scoring.compute_score("example", 1, method="tfidf")
        self.assertGreater(score, 0, "TFIDF score for 'example' in doc 1 should be greater than 0.")

        score = self.scoring.compute_score("test", 2, method="tfidf")
        self.assertGreater(score, 0, "TFIDF score for 'test' in doc 2 should be greater than 0.")

        score = self.scoring.compute_score("search", 3, method="tfidf")
        self.assertGreater(score, 0, "TFIDF score for 'search' in doc 3 should be greater than 0.")

    def test_bm25_scoring(self):
        """Test BM25 scoring for various terms and documents."""
        score = self.scoring.compute_score("example", 1, method="bm25")
        self.assertGreater(score, 0, "BM25 score for 'example' in doc 1 should be greater than 0.")

        score = self.scoring.compute_score("test", 2, method="bm25")
        self.assertGreater(score, 0, "BM25 score for 'test' in doc 2 should be greater than 0.")

        score = self.scoring.compute_score("search", 3, method="bm25")
        self.assertGreater(score, 0, "BM25 score for 'search' in doc 3 should be greater than 0.")

    def test_nonexistent_term(self):
        """Test scoring for a term that does not exist in the lexicon."""
        score = self.scoring.compute_score("nonexistent", 1, method="tfidf")
        self.assertEqual(score, 0, "TFIDF score for a nonexistent term should be 0.")

        score = self.scoring.compute_score("nonexistent", 1, method="bm25")
        self.assertEqual(score, 0, "BM25 score for a nonexistent term should be 0.")

if __name__ == "__main__":
    unittest.main()
