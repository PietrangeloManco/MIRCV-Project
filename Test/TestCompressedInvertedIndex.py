import unittest
import os
from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex


class TestCompressedInvertedIndex(unittest.TestCase):

    def setUp(self):
        """Set up a sample CompressedInvertedIndex for testing."""
        self.index = CompressedInvertedIndex()

        # Sample data for compression
        self.term = "test"
        self.doc_ids = [1, 2, 3]
        self.frequencies = [5, 10, 15]

        # Compress and add the postings
        self.index.compress_and_add_postings(self.term, self.doc_ids, self.frequencies)

        # Test file names
        self.compressed_file = "test_compressed_index.bin"
        self.text_file = "test_index.txt"

    def tearDown(self):
        """Clean up test files after each test."""
        if os.path.exists(self.compressed_file):
            os.remove(self.compressed_file)
        if os.path.exists(self.text_file):
            os.remove(self.text_file)

    def test_add_and_get_compressed_postings(self):
        """Test adding and retrieving compressed postings."""
        compressed_postings = self.index.get_compressed_postings(self.term)
        self.assertGreater(len(compressed_postings), 0)  # Ensure that there is compressed data
        self.assertIsInstance(compressed_postings, bytes)  # Ensure that the data is in bytes

    def test_get_uncompressed_postings(self):
        """Test fetching and decompressing postings for a given term."""
        postings = self.index.get_uncompressed_postings(self.term)
        self.assertEqual(len(postings), 3)  # We added 3 postings for this term
        self.assertEqual(postings[0].doc_id, 1)
        self.assertEqual(postings[0].payload, 5)
        self.assertEqual(postings[1].doc_id, 2)
        self.assertEqual(postings[1].payload, 10)
        self.assertEqual(postings[2].doc_id, 3)
        self.assertEqual(postings[2].payload, 15)

    def test_write_and_load_compressed_index(self):
        """Test writing the compressed index to a file and loading it back."""
        # Write the compressed index to a file
        self.index.write_compressed_index_to_file(self.compressed_file)

        # Load the compressed index from the file
        loaded_index = CompressedInvertedIndex.load_compressed_index_to_memory(self.compressed_file)

        # Check if the loaded index contains the correct term
        compressed_data = loaded_index.get_compressed_postings(self.term)
        self.assertGreater(len(compressed_data), 0)  # Ensure that the data exists

        # Test decompression of the loaded data
        postings = loaded_index.get_uncompressed_postings(self.term)
        self.assertEqual(len(postings), 3)  # We added 3 postings for this term
        self.assertEqual(postings[0].doc_id, 1)
        self.assertEqual(postings[0].payload, 5)
        self.assertEqual(postings[1].doc_id, 2)
        self.assertEqual(postings[1].payload, 10)
        self.assertEqual(postings[2].doc_id, 3)
        self.assertEqual(postings[2].payload, 15)

    def test_compress_and_add_postings(self):
        """Test compressing and adding postings to the index."""
        term = "example"
        doc_ids = [4, 5, 6]
        frequencies = [20, 25, 30]

        # Compress and add new postings
        self.index.compress_and_add_postings(term, doc_ids, frequencies)

        # Check if the compressed data exists for the new term
        compressed_data = self.index.get_compressed_postings(term)
        self.assertGreater(len(compressed_data), 0)  # Ensure that compressed data exists

        # Test decompression of the new term's data
        postings = self.index.get_uncompressed_postings(term)
        self.assertEqual(len(postings), 3)  # We added 3 postings
        self.assertEqual(postings[0].doc_id, 4)
        self.assertEqual(postings[0].payload, 20)
        self.assertEqual(postings[1].doc_id, 5)
        self.assertEqual(postings[1].payload, 25)
        self.assertEqual(postings[2].doc_id, 6)
        self.assertEqual(postings[2].payload, 30)

    def test_get_terms(self):
        """Test fetching all terms from the compressed inverted index."""
        term = "example"
        doc_ids = [4, 5, 6]
        frequencies = [20, 25, 30]

        # Compress and add new postings
        self.index.compress_and_add_postings(term, doc_ids, frequencies)

        terms = list(self.index.get_terms())
        self.assertIn("test", terms)
        self.assertIn("example", terms)

if __name__ == "__main__":
    unittest.main()
