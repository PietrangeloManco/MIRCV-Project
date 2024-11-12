import unittest
import os

from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex


class TestCompressedInvertedIndex(unittest.TestCase):
    def setUp(self):
        self.index = CompressedInvertedIndex(chunk_size=3)  # Small chunk size for testing

        # Test data
        self.term1 = "apple"
        self.doc_ids1 = [1, 4, 7, 10, 13]
        self.freqs1 = [2, 1, 3, 1, 2]

        self.term2 = "banana"
        self.doc_ids2 = [2, 5, 8, 11, 14]
        self.freqs2 = [1, 2, 1, 3, 1]

    def tearDown(self):
        # Clean up any test files
        if os.path.exists("test_index.bin"):
            os.remove("test_index.bin")

    def test_add_and_retrieve_postings(self):
        # Add postings
        self.index.compress_and_add_postings(self.term1, self.doc_ids1, self.freqs1)

        # Test retrieval
        postings = self.index.get_uncompressed_postings(self.term1)

        # Verify results
        self.assertEqual(len(postings), len(self.doc_ids1))
        for i, posting in enumerate(postings):
            self.assertEqual(posting.doc_id, self.doc_ids1[i])
            self.assertEqual(posting.payload, self.freqs1[i])

    def test_multiple_terms(self):
        # Add multiple terms
        self.index.compress_and_add_postings(self.term1, self.doc_ids1, self.freqs1)
        self.index.compress_and_add_postings(self.term2, self.doc_ids2, self.freqs2)

        # Test retrieval for both terms
        terms = list(self.index.get_terms())
        self.assertEqual(len(terms), 2)
        self.assertIn(self.term1, terms)
        self.assertIn(self.term2, terms)

    def test_posting_list_retrieval(self):
        # Add postings
        self.index.compress_and_add_postings(self.term1, self.doc_ids1, self.freqs1)

        # Get posting list
        posting_list = self.index.get_posting_list(self.term1)

        # Verify posting list properties
        self.assertIsNotNone(posting_list)
        self.assertEqual(len(posting_list.compressed_chunks), 2)  # With chunk_size=3, should have 2 chunks
        self.assertEqual(len(posting_list.chunk_boundaries), 2)

        # Verify boundaries
        self.assertEqual(posting_list.chunk_boundaries[0][0], 1)  # First doc_id in first chunk
        self.assertEqual(posting_list.chunk_boundaries[1][1], 13)  # Last doc_id in last chunk

    def test_file_persistence(self):
        # Add some postings
        self.index.compress_and_add_postings(self.term1, self.doc_ids1, self.freqs1)
        self.index.compress_and_add_postings(self.term2, self.doc_ids2, self.freqs2)

        # Write to file
        test_filename = "test_index.bin"
        self.index.write_compressed_index_to_file(test_filename)

        # Load from file
        loaded_index = CompressedInvertedIndex.load_compressed_index_to_memory(test_filename)

        # Verify loaded data
        terms = list(loaded_index.get_terms())
        self.assertEqual(len(terms), 2)

        # Check postings for term1
        postings1 = loaded_index.get_uncompressed_postings(self.term1)
        self.assertEqual(len(postings1), len(self.doc_ids1))
        for i, posting in enumerate(postings1):
            self.assertEqual(posting.doc_id, self.doc_ids1[i])
            self.assertEqual(posting.payload, self.freqs1[i])

        # Check postings for term2
        postings2 = loaded_index.get_uncompressed_postings(self.term2)
        self.assertEqual(len(postings2), len(self.doc_ids2))
        for i, posting in enumerate(postings2):
            self.assertEqual(posting.doc_id, self.doc_ids2[i])
            self.assertEqual(posting.payload, self.freqs2[i])

    def test_nonexistent_term(self):
        # Test retrieval of non-existent term
        postings = self.index.get_uncompressed_postings("nonexistent")
        self.assertEqual(len(postings), 0)

        posting_list = self.index.get_posting_list("nonexistent")
        self.assertIsNone(posting_list)

    def test_merge_postings(self):
        # Add initial postings
        self.index.compress_and_add_postings(self.term1, [1, 4, 7], [1, 2, 3])

        # Add more postings for the same term
        self.index.compress_and_add_postings(self.term1, [10, 13], [4, 5])

        # Verify merged postings
        postings = self.index.get_uncompressed_postings(self.term1)
        expected_doc_ids = [1, 4, 7, 10, 13]
        expected_freqs = [1, 2, 3, 4, 5]

        self.assertEqual(len(postings), len(expected_doc_ids))
        for i, posting in enumerate(postings):
            self.assertEqual(posting.doc_id, expected_doc_ids[i])
            self.assertEqual(posting.payload, expected_freqs[i])


if __name__ == '__main__':
    unittest.main()