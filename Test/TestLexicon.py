import os
import unittest

from Lexicon import Lexicon


class TestLexicon(unittest.TestCase):

    def setUp(self):
        """For each test, create a lexicon and a temporary file path to eventually store it."""
        self.lexicon = Lexicon.Lexicon()
        self.temp_file = "test_lexicon.txt"

    def tearDown(self):
        """Delete the temporary lexicon after each test."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

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

        self.lexicon.write_to_file(self.temp_file)

        # Load the lexicon from the file
        loaded_lexicon = Lexicon.Lexicon.load_from_file(self.temp_file)

        # Verify the terms and frequencies
        self.assertEqual(loaded_lexicon.get_term_info("apple"), 5)
        self.assertEqual(loaded_lexicon.get_term_info("banana"), 3)
        self.assertIsNone(loaded_lexicon.get_term_info("cherry"))

    def test_empty_lexicon(self):
        """Test behavior with an empty lexicon."""
        self.assertEqual(self.lexicon.get_all_terms(), [])
        self.assertIsNone(self.lexicon.get_term_info("nonexistent"))


if __name__ == "__main__":
    unittest.main()
