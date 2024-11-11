import os
import unittest
from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from InvertedIndex.Merger import Merger # Assuming the Merger class is in Merger.py
from Utils.CompressionTools import CompressionTools


class TestMerger(unittest.TestCase):

    def setUp(self):
        """Set up test data and actual objects."""
        self.merger = Merger()

        # Create some test terms and postings data
        self.term1 = "apple"
        self.term2 = "banana"

        # Let's create some simple postings data (doc_id, frequency)
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

        # Save paths for the indices (we will assume they are saved and can be loaded)
        index_paths = ["index1", "index2", "index3"]

        # Save the indices to memory, for testing purposes
        CompressedInvertedIndex.write_compressed_index_to_file(self.index1, "index1")
        CompressedInvertedIndex.write_compressed_index_to_file(self.index2, "index2")
        CompressedInvertedIndex.write_compressed_index_to_file(index3, "index3")

        # Merge the indices
        final_index = self.merger.merge_multiple_compressed_indices(index_paths)

        # Check that all terms are present in the final merged index
        self.assertTrue(self.term1 in final_index.get_terms())
        self.assertTrue(self.term2 in final_index.get_terms())
        self.assertTrue(term3 in final_index.get_terms())

        for i in range (1, 4):
            os.remove(f"index{i}")


    def test_merge_empty_indices(self):
        """Test the case when no indices are provided."""
        with self.assertRaises(ValueError):
            self.merger.merge_multiple_compressed_indices([])

    def test_merge_single_index(self):
        """Test merging a single index (no merging actually happens)."""
        CompressedInvertedIndex.write_compressed_index_to_file(self.index1, "index1")
        index_paths = ["index1"]

        # Merge a single index
        final_index = self.merger.merge_multiple_compressed_indices(index_paths)

        # Iterate over each term in final_index and assert that the postings match with self.index1
        for term in final_index.get_terms():
            # Get uncompressed postings for the term from both final_index and self.index1
            final_postings = final_index.get_uncompressed_postings(term)
            original_postings = self.index1.get_uncompressed_postings(term)

            # Assert that the postings are equal for the term
            self.assertEqual(final_postings, original_postings, f"Postings for term '{term}' do not match.")
        os.remove("index1")

if __name__ == "__main__":
    unittest.main()
