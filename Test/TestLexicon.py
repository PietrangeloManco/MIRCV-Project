import unittest
import os

from InvertedIndex.InvertedIndex import InvertedIndex
from Lexicon.Lexicon import Lexicon


class TestLexicon(unittest.TestCase):

    def setUp(self):
        """Setup for the tests."""
        self.lexicon = Lexicon()

    def test_add_term_to_lexicon(self):
        """Test adding a term to the lexicon."""
        # Add a term to the lexicon with some metadata
        self.lexicon.add_term("apple", position=100, term_frequency=5)

        # Verify the term has been added
        term_info = self.lexicon.get_term_info("apple")
        self.assertIsNotNone(term_info)
        self.assertEqual(term_info["position"], 100)
        self.assertEqual(term_info["term_frequency"], 5)

    def test_get_all_terms_from_lexicon(self):
        """Test retrieving all terms from the lexicon."""
        # Add some terms to the lexicon
        self.lexicon.add_term("apple", position=100, term_frequency=5)
        self.lexicon.add_term("banana", position=150, term_frequency=3)

        # Get all terms and verify
        terms = self.lexicon.get_all_terms()
        self.assertIn("apple", terms)
        self.assertIn("banana", terms)

    def test_write_and_load_from_file(self):
        """Test writing and loading the lexicon to and from a file."""
        # Add some terms to the lexicon
        self.lexicon.add_term("apple", position=100, term_frequency=5)
        self.lexicon.add_term("banana", position=150, term_frequency=3)

        # Write the lexicon to a file
        lexicon_file = "test_lexicon.txt"
        self.lexicon.write_to_file(lexicon_file)

        # Load the lexicon from the file
        loaded_lexicon = Lexicon.load_from_file(lexicon_file)

        # Verify that the loaded lexicon contains the correct data
        apple_info = loaded_lexicon.get_term_info("apple")
        banana_info = loaded_lexicon.get_term_info("banana")

        self.assertEqual(apple_info["position"], 100)
        self.assertEqual(apple_info["term_frequency"], 5)

        self.assertEqual(banana_info["position"], 150)
        self.assertEqual(banana_info["term_frequency"], 3)

        # Clean up
        os.remove(lexicon_file)

    def test_build_lexicon_from_inverted_index(self):
        """Test building the lexicon from an inverted index."""
        # Simulate an inverted index with postings
        inverted_index = InvertedIndex()
        inverted_index.add_posting("apple", 1, 5)
        inverted_index.add_posting("apple", 2, 3)
        inverted_index.add_posting("banana", 3, 2)

        # Build the lexicon from the inverted index
        self.lexicon.build_lexicon(inverted_index)

        # Check the lexicon for correct term frequencies
        apple_info = self.lexicon.get_term_info("apple")
        banana_info = self.lexicon.get_term_info("banana")

        self.assertEqual(apple_info["term_frequency"], 2)  # 2 postings for "apple"
        self.assertEqual(banana_info["term_frequency"], 1)  # 1 posting for "banana"

    def test_empty_lexicon(self):
        """Test the lexicon when it's empty."""
        terms = self.lexicon.get_all_terms()
        self.assertEqual(terms, [])
        term_info = self.lexicon.get_term_info("apple")
        self.assertIsNone(term_info)


if __name__ == "__main__":
    unittest.main()
