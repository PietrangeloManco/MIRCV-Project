from typing import List, Optional, Tuple
from InvertedIndex.Posting import Posting
from Utils.CompressionTools import CompressionTools

class PostingList:
    def __init__(self, compressed_chunks: List[bytes], chunk_boundaries: List[Tuple[int, int]],
                 skip_pointers: List[int]):
        """
        Initializes the PostingList with compressed chunks and their boundaries.

        :param compressed_chunks: List of compressed byte arrays, each representing a chunk of postings
        :param chunk_boundaries: List of tuples (start_doc_id, end_doc_id) for each chunk
        :param skip_pointers: List of doc_ids that serve as skip pointers
        """
        self.compressed_chunks = compressed_chunks
        self.chunk_boundaries = chunk_boundaries
        self.skip_pointers = skip_pointers

        # Current state
        self.current_chunk_index = 0
        self.current_posting_index = 0
        self.current_chunk_doc_ids = []
        self.current_chunk_frequencies = []

        # Load first chunk if available
        if compressed_chunks:
            self._decompress_chunk(0)

    def _decompress_chunk(self, chunk_index: int) -> None:
        """
        Decompresses a specific chunk of the posting list.

        :param chunk_index: Index of the chunk to decompress
        """
        if 0 <= chunk_index < len(self.compressed_chunks):
            self.current_chunk_doc_ids, self.current_chunk_frequencies = CompressionTools.p_for_delta_decompress(
                self.compressed_chunks[chunk_index]
            )
            self.current_chunk_index = chunk_index
            self.current_posting_index = 0

    def _find_chunk_for_doc_id(self, target_doc_id: int) -> Optional[int]:
        """
        Binary search to find the chunk that might contain the target_doc_id.

        :param target_doc_id: The doc_id to search for
        :return: Index of the chunk that might contain the doc_id, or None if not found
        """
        if not self.chunk_boundaries:  # Handle empty list case
            return None

        left, right = 0, len(self.chunk_boundaries) - 1

        while left <= right:
            mid = (left + right) // 2
            start_doc_id, end_doc_id = self.chunk_boundaries[mid]

            if start_doc_id <= target_doc_id <= end_doc_id:
                return mid
            elif target_doc_id < start_doc_id:
                right = mid - 1
            else:
                left = mid + 1

        # If we didn't find an exact match, return the first chunk where target_doc_id
        # is less than the chunk's start_doc_id
        for i, (start_doc_id, _) in enumerate(self.chunk_boundaries):
            if target_doc_id < start_doc_id:
                return i
        return None

    def next_geq(self, target_doc_id: int) -> Optional[Posting]:
        """
        Finds the next Posting with a doc_id >= target_doc_id.

        :param target_doc_id: The doc_id to search for
        :return: The next Posting object with doc_id >= target_doc_id, or None if not found
        """
        if not self.compressed_chunks:  # Handle empty list case
            return None

        target_chunk = self._find_chunk_for_doc_id(target_doc_id)

        if target_chunk is None:
            # If target_doc_id is beyond our last chunk, return None
            if self.chunk_boundaries and target_doc_id > self.chunk_boundaries[-1][1]:
                return None
            # If target_doc_id is before our first chunk, use the first chunk
            target_chunk = 0

        # If we need to switch chunks, decompress the new chunk
        if target_chunk != self.current_chunk_index:
            self._decompress_chunk(target_chunk)
            self.current_posting_index = 0  # Reset posting index when switching chunks

        # Search within the current chunk
        while self.current_posting_index < len(self.current_chunk_doc_ids):
            current_doc_id = self.current_chunk_doc_ids[self.current_posting_index]

            if current_doc_id >= target_doc_id:
                posting = Posting(
                    current_doc_id,
                    self.current_chunk_frequencies[self.current_posting_index]
                )
                return posting

            self.current_posting_index += 1

        # If we reach end of chunk, try next chunk
        if target_chunk + 1 < len(self.compressed_chunks):
            self._decompress_chunk(target_chunk + 1)
            self.current_posting_index = 0  # Reset posting index
            # Return the first posting in the next chunk
            if self.current_chunk_doc_ids:
                return Posting(
                    self.current_chunk_doc_ids[0],
                    self.current_chunk_frequencies[0]
                )

        return None

    def reset(self) -> None:
        """Resets the iterator to the beginning of the posting list."""
        self.current_chunk_index = 0
        self.current_posting_index = 0
        if self.compressed_chunks:
            self._decompress_chunk(0)