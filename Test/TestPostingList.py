import unittest
from typing import List, Tuple

from InvertedIndex.PostingList import PostingList
from Utils.CompressionTools import CompressionTools


class TestPostingList(unittest.TestCase):
    @staticmethod
    def create_test_chunks(doc_ids: List[int], freqs: List[int], chunk_size: int) -> Tuple[
        List[bytes], List[Tuple[int, int]]]:
        """Helper method to create test chunks from doc_ids and frequencies"""
        chunks = []
        boundaries = []

        for i in range(0, len(doc_ids), chunk_size):
            chunk_doc_ids = doc_ids[i:i + chunk_size]
            chunk_freqs = freqs[i:i + chunk_size]

            compressed_chunk = CompressionTools.p_for_delta_compress(chunk_doc_ids, chunk_freqs)
            chunks.append(compressed_chunk)
            boundaries.append((chunk_doc_ids[0], chunk_doc_ids[-1]))

        return chunks, boundaries

    def setUp(self):
        # Create test data with clear chunk boundaries
        self.doc_ids = [1, 2, 3,  # chunk 0
                        5, 7, 9,  # chunk 1
                        12, 15, 18,  # chunk 2
                        20, 25, 30]  # chunk 3
        self.frequencies = [1, 2, 1,
                            3, 2, 1,
                            4, 2, 1,
                            2, 3, 1]
        self.chunk_size = 3

        # Create chunks and boundaries
        self.chunks, self.boundaries = self.create_test_chunks(
            self.doc_ids,
            self.frequencies,
            self.chunk_size
        )

        # Create skip pointers (every other chunk)
        self.skip_pointers = [self.doc_ids[i] for i in range(0, len(self.doc_ids), self.chunk_size * 2)]

        # Initialize PostingList
        self.posting_list = PostingList(
            compressed_chunks=self.chunks,
            chunk_boundaries=self.boundaries,
            skip_pointers=self.skip_pointers
        )

    def test_initialization(self):
        """Test if PostingList initializes correctly"""
        self.assertEqual(len(self.posting_list.compressed_chunks), 4)  # 12 items / 3 per chunk = 4 chunks
        self.assertEqual(len(self.posting_list.chunk_boundaries), 4)
        self.assertEqual(self.posting_list.current_chunk_index, 0)
        self.assertEqual(self.posting_list.current_posting_index, 0)

        # Test if first chunk is decompressed correctly
        self.assertEqual(self.posting_list.current_chunk_doc_ids, [1, 2, 3])
        self.assertEqual(self.posting_list.current_chunk_frequencies, [1, 2, 1])

    def test_find_chunk_for_doc_id(self):
        """Test binary search for finding correct chunk"""
        # Test finding exact doc_ids
        self.assertEqual(self.posting_list._find_chunk_for_doc_id(1), 0)  # First chunk
        self.assertEqual(self.posting_list._find_chunk_for_doc_id(7), 1)  # Second chunk
        self.assertEqual(self.posting_list._find_chunk_for_doc_id(15), 2)  # Third chunk

        # Test doc_ids that fall between chunks
        self.assertEqual(self.posting_list._find_chunk_for_doc_id(4), 1)  # Should return next chunk
        self.assertEqual(self.posting_list._find_chunk_for_doc_id(10), 2)

        # Test boundary cases
        self.assertEqual(self.posting_list._find_chunk_for_doc_id(0), 0)  # Before first doc_id
        self.assertIsNone(self.posting_list._find_chunk_for_doc_id(31))  # After last doc_id

    def test_next_geq_basic(self):
        """Test basic next_geq functionality"""
        # Test exact matches
        posting = self.posting_list.next_geq(1)
        self.assertEqual(posting.doc_id, 1)
        self.assertEqual(posting.payload, 1)

        posting = self.posting_list.next_geq(15)
        self.assertEqual(posting.doc_id, 15)
        self.assertEqual(posting.payload, 2)

    def test_next_geq_between_values(self):
        """Test next_geq with values between actual doc_ids"""
        # Should return next higher doc_id
        posting = self.posting_list.next_geq(4)
        self.assertEqual(posting.doc_id, 5)
        self.assertEqual(posting.payload, 3)

        posting = self.posting_list.next_geq(10)
        self.assertEqual(posting.doc_id, 12)
        self.assertEqual(posting.payload, 4)

    def test_next_geq_chunk_boundaries(self):
        """Test next_geq at chunk boundaries"""
        # Get last doc_id of first chunk
        posting = self.posting_list.next_geq(3)
        self.assertEqual(posting.doc_id, 3)

        # Get first doc_id of next chunk
        posting = self.posting_list.next_geq(4)
        self.assertEqual(posting.doc_id, 5)

    def test_next_geq_out_of_range(self):
        """Test next_geq with out-of-range values"""
        # Test before first doc_id
        posting = self.posting_list.next_geq(0)
        self.assertEqual(posting.doc_id, 1)

        # Test after last doc_id
        posting = self.posting_list.next_geq(31)
        self.assertIsNone(posting)

    def test_empty_posting_list(self):
        """Test behavior with empty posting list"""
        empty_posting_list = PostingList([], [], [])
        self.assertIsNone(empty_posting_list.next_geq(1))
        self.assertIsNone(empty_posting_list._find_chunk_for_doc_id(1))

        # Test reset on empty list
        empty_posting_list.reset()  # Should not raise any errors
        self.assertEqual(empty_posting_list.current_chunk_index, 0)
        self.assertEqual(len(empty_posting_list.current_chunk_doc_ids), 0)


if __name__ == '__main__':
    unittest.main()