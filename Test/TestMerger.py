import os
import unittest

from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from InvertedIndex.Merger import Merger  # Assuming the Merger class is in Merger.py
from Utils.CompressionTools import CompressionTools


class TestMerger(unittest.TestCase):

    def setUp(self):
        """Set up test data and actual objects."""
        self.merger = Merger()

        # Track test files to clean up later
        self.test_files = []

        # Test terms
        self.term1 = "apple"
        self.term2 = "banana"

        # Simple postings data (doc_id, frequency)
        doc_ids1 = [1, 2, 3]
        frequencies1 = [1, 2, 3]
        doc_ids2 = [2, 3, 4]
        frequencies2 = [4, 5, 6]

        # Compress the postings using PForDelta compression
        self.postings1 = CompressionTools.p_for_delta_compress(doc_ids1, frequencies1)
        self.postings2 = CompressionTools.p_for_delta_compress(doc_ids2, frequencies2)

        # Create two instances of CompressedInvertedIndex
        self.index1 = CompressedInvertedIndex()
        self.index2 = CompressedInvertedIndex()

        # Add some terms with their corresponding compressed postings
        self.index1.add_compressed_postings(self.term1, self.postings1)
        self.index1.add_compressed_postings(self.term2, self.postings2)
        self.index2.add_compressed_postings(self.term1, self.postings2)  # term1 also in index2

    def tearDown(self):
        """Clean up any files created during testing."""
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_merge_compressed_postings(self):
        """Test the merging of compressed postings."""
        # Merge the postings for term1 (which appears in both index1 and index2)
        merged_postings = self.merger._merge_compressed_postings(self.postings1, self.postings2)

        # Decompress the merged postings to check if they are correct
        doc_ids, frequencies = CompressionTools.p_for_delta_decompress(merged_postings)

        # Check that the doc_ids are merged correctly
        self.assertEqual(doc_ids, [1, 2, 3, 4])  # Combined doc_ids from both postings

        # Check that the frequencies are summed correctly
        self.assertEqual(frequencies, [1, 6, 8, 6])  # Frequencies should be summed where doc_ids are the same

    def test_merge_two_indices(self):
        """Test merging two indices."""
        # Merge the two indices
        merged_index = self.merger._merge_two_indices(self.index1, self.index2)

        # Check if the merged index contains both terms
        self.assertTrue(self.term1 in merged_index.get_terms())
        self.assertTrue(self.term2 in merged_index.get_terms())

        # Check that the postings for term1 are correctly merged
        merged_postings = merged_index.get_compressed_postings(self.term1)
        doc_ids, frequencies = CompressionTools.p_for_delta_decompress(merged_postings)

        # Verify that doc_ids are merged and frequencies summed
        self.assertEqual(doc_ids, [1, 2, 3, 4])
        self.assertEqual(frequencies, [1, 6, 8, 6])

    def test_merge_multiple_compressed_indices(self):
        """Test merging multiple indices."""
        # Create some additional test data for the third index
        term3 = "cherry"
        doc_ids3 = [3, 4, 5]
        frequencies3 = [7, 8, 9]
        postings3 = CompressionTools.p_for_delta_compress(doc_ids3, frequencies3)

        # Create a third index
        index3 = CompressedInvertedIndex()
        index3.add_compressed_postings(term3, postings3)

        # Save paths for the indices
        index_paths = ["index1", "index2", "index3"]

        # Save the indices to files and track the filenames
        for idx, index in enumerate([self.index1, self.index2, index3], start=1):
            file_name = f"index{idx}"
            index.write_compressed_index_to_file(file_name)
            self.test_files.append(file_name)

        # Merge the indices
        final_index = self.merger.merge_multiple_compressed_indices(index_paths)

        # Check that all terms are present in the final merged index
        self.assertTrue(self.term1 in final_index.get_terms())
        self.assertTrue(self.term2 in final_index.get_terms())
        self.assertTrue(term3 in final_index.get_terms())

    def test_merge_empty_indices(self):
        """Test the case when no indices are provided."""
        with self.assertRaises(ValueError):
            self.merger.merge_multiple_compressed_indices([])

    def test_merge_single_index(self):
        """Test merging a single index (no merging actually happens)."""
        file_name = "index1"
        self.index1.write_compressed_index_to_file(file_name)
        self.test_files.append(file_name)

        index_paths = [file_name]

        # Merge a single index
        final_index = self.merger.merge_multiple_compressed_indices(index_paths)

        # Verify terms in the final index
        self.assertEqual(
            set(final_index.get_terms()),
            set(self.index1.get_terms()),
            "Terms in the final index do not match the original index."
        )

        # Compare the postings for each term
        for term in final_index.get_terms():
            final_postings = final_index.get_uncompressed_postings(term)
            original_postings = self.index1.get_uncompressed_postings(term)

            # Extract doc_id and frequency from the postings
            final_postings_data = [(p.doc_id, p.payload) for p in final_postings]
            original_postings_data = [(p.doc_id, p.payload) for p in original_postings]

            # Debugging: Log mismatched postings
            if final_postings_data != original_postings_data:
                print(f"Mismatch for term '{term}':")
                print(f"Final postings data: {final_postings_data}")
                print(f"Original postings data: {original_postings_data}")

            # Assert that the data matches
            self.assertEqual(
                final_postings_data,
                original_postings_data,
                f"Postings data for term '{term}' do not match."
            )


if __name__ == "__main__":
    unittest.main()
