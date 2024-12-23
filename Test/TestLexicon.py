import os
import unittest

from Lexicon import Lexicon


class TestLexicon(unittest.TestCase):
    def setUp(self):
        """Set up a fresh instance of Lexicon before each test."""
        self.lexicon = Lexicon.Lexicon()

    def test_add_and_get_term(self):
        """Test adding terms and retrieving their document frequency."""
        self.lexicon.add_term("apple", 5)
        self.lexicon.add_term("banana", 3)

        # Verify the document frequencies
        self.assertEqual(self.lexicon.get_term_info("apple"), 5)
        self.assertEqual(self.lexicon.get_term_info("banana"), 3)
        self.assertIsNone(self.lexicon.get_term_info("cherry"))  # Non-existent term

    def test_get_all_terms(self):
        """Test retrieving all terms in the lexicon."""
        self.lexicon.add_term("apple", 5)
        self.lexicon.add_term("banana", 3)

        terms = self.lexicon.get_all_terms()
        self.assertIn("apple", terms)
        self.assertIn("banana", terms)
        self.assertEqual(len(terms), 2)

    def test_write_and_load_from_file(self):
        """Test saving the lexicon to a file and loading it back."""
        self.lexicon.add_term("apple", 5)
        self.lexicon.add_term("banana", 3)

        # Write to a temporary file
        temp_file = "test_lexicon.txt"
        self.lexicon.write_to_file(temp_file)

        # Load the lexicon from the file
        loaded_lexicon = Lexicon.Lexicon.load_from_file(temp_file)

        # Verify the terms and frequencies
        self.assertEqual(loaded_lexicon.get_term_info("apple"), 5)
        self.assertEqual(loaded_lexicon.get_term_info("banana"), 3)
        self.assertIsNone(loaded_lexicon.get_term_info("cherry"))

        # Clean up temporary file
        os.remove(temp_file)

    def test_build_lexicon(self):
        """Test building the lexicon from an inverted index."""

        class DummyInvertedIndex:
            def __init__(self):
                self.index = {
                    "apple": [(1, 2), (2, 3)],
                    "banana": [(1, 1)],
                    "cherry": [(2, 1), (3, 2), (4, 1)]
                }

            def get_terms(self):
                return self.index.keys()

            def get_postings(self, term):
                return self.index[term]

        inverted_index = DummyInvertedIndex()
        self.lexicon.build_lexicon(inverted_index)

        # Verify the built lexicon
        self.assertEqual(self.lexicon.get_term_info("apple"), {"term_frequency": 2})
        self.assertEqual(self.lexicon.get_term_info("banana"), {"term_frequency": 1})
        self.assertEqual(self.lexicon.get_term_info("cherry"), {"term_frequency": 3})

    def test_empty_lexicon(self):
        """Test behavior with an empty lexicon."""
        self.assertEqual(self.lexicon.get_all_terms(), [])
        self.assertIsNone(self.lexicon.get_term_info("nonexistent"))

    def tearDown(self):
        """Clean up after each test if necessary."""
        pass


if __name__ == "__main__":
    unittest.main()
