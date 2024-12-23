import os
import unittest

from src.InvertedIndex.InvertedIndex import InvertedIndex


class TestInvertedIndex(unittest.TestCase):

    def setUp(self):
        """Set up a sample InvertedIndex for testing."""
        self.index = InvertedIndex()
        self.index.add_posting("test", 1, 5)
        self.index.add_posting("test", 2, 10)
        self.index.add_posting("example", 3, 15)

        # Test file names
        self.text_file = "test_index.txt"
        self.compressed_file = "test_compressed_index.bin"

    def tearDown(self):
        """Clean up test files after each test."""
        if os.path.exists(self.text_file):
            os.remove(self.text_file)
        if os.path.exists(self.compressed_file):
            os.remove(self.compressed_file)

    def test_add_and_get_postings(self):
        """Test adding postings and retrieving them."""
        postings = self.index.get_postings("test")
        self.assertEqual(len(postings), 2)
        self.assertEqual(postings[0].doc_id, 1)
        self.assertEqual(postings[0].payload, 5)
        self.assertEqual(postings[1].doc_id, 2)
        self.assertEqual(postings[1].payload, 10)

    def test_get_terms(self):
        """Test retrieving unique terms from the index."""
        terms = self.index.get_terms()
        self.assertEqual(set(terms), {"test", "example"})

    def test_write_and_load_from_file(self):
        """Test writing the index to a file and loading it back."""
        self.index.write_to_file(self.text_file)
        loaded_index = InvertedIndex.load_from_file(self.text_file)

        postings = loaded_index.get_postings("test")
        self.assertEqual(len(postings), 2)
        self.assertEqual(postings[0].doc_id, 1)
        self.assertEqual(postings[0].payload, 5)
        self.assertEqual(postings[1].doc_id, 2)
        self.assertEqual(postings[1].payload, 10)

        postings_example = loaded_index.get_postings("example")
        self.assertEqual(len(postings_example), 1)
        self.assertEqual(postings_example[0].doc_id, 3)
        self.assertEqual(postings_example[0].payload, 15)

    def test_compression_and_decompression(self):
        """Test writing and loading a compressed index."""
        # Write the index to a compressed file
        self.index.write_index_compressed_to_file(self.compressed_file)

        # Load the index from the compressed file
        loaded_index = InvertedIndex.load_compressed_index_from_file(self.compressed_file)

        postings = loaded_index.get_postings("test")
        self.assertEqual(len(postings), 2)
        self.assertEqual(postings[0].doc_id, 1)
        self.assertEqual(postings[0].payload, 5)
        self.assertEqual(postings[1].doc_id, 2)
        self.assertEqual(postings[1].payload, 10)

        postings_example = loaded_index.get_postings("example")
        self.assertEqual(len(postings_example), 1)
        self.assertEqual(postings_example[0].doc_id, 3)
        self.assertEqual(postings_example[0].payload, 15)


if __name__ == "__main__":
    unittest.main()
