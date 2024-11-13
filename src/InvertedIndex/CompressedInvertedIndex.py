import struct
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from InvertedIndex.Posting import Posting
from InvertedIndex.PostingList import PostingList
from Utils.CompressionTools import CompressionTools


@dataclass
class ChunkedPostings:
    """Helper class to store chunked posting data"""
    chunks: List[bytes]
    boundaries: List[Tuple[int, int]]


class CompressedInvertedIndex:
    def __init__(self, chunk_size: int = 1000):
        """
        Initialize the compressed inverted index with chunked postings.

        :param chunk_size: Number of postings per chunk
        """
        self._compressed_index: Dict[str, ChunkedPostings] = {}
        self.chunk_size = chunk_size

    def _create_posting_chunks(self, doc_ids: List[int], frequencies: List[int]) -> ChunkedPostings:
        """
        Create compressed chunks from doc_ids and frequencies.

        :param doc_ids: List of document IDs
        :param frequencies: List of term frequencies
        :return: ChunkedPostings containing compressed chunks and their boundaries
        """
        chunks = []
        boundaries = []

        for i in range(0, len(doc_ids), self.chunk_size):
            chunk_doc_ids = doc_ids[i:i + self.chunk_size]
            chunk_freqs = frequencies[i:i + self.chunk_size]

            # Compress the chunk
            compressed_chunk = CompressionTools.p_for_delta_compress(chunk_doc_ids, chunk_freqs)

            # Store chunk boundaries for binary search
            boundaries.append((chunk_doc_ids[0], chunk_doc_ids[-1]))
            chunks.append(compressed_chunk)

        return ChunkedPostings(chunks=chunks, boundaries=boundaries)

    def write_compressed_index_to_file(self, filename: str) -> None:
        """Writes the chunked inverted index to a file."""
        with open(filename, 'wb') as f:
            for term, chunked_postings in self._compressed_index.items():
                # Write term
                term_bytes = term.encode('utf-8')
                f.write(struct.pack("H", len(term_bytes)))
                f.write(term_bytes)

                # Write number of chunks
                f.write(struct.pack("I", len(chunked_postings.chunks)))

                # Write boundaries
                for start_id, end_id in chunked_postings.boundaries:
                    f.write(struct.pack("II", start_id, end_id))

                # Write chunks
                for chunk in chunked_postings.chunks:
                    f.write(struct.pack("I", len(chunk)))
                    f.write(chunk)

    @staticmethod
    def load_compressed_index_to_memory(filename: str) -> 'CompressedInvertedIndex':
        """Loads a chunked compressed inverted index from file."""
        index = CompressedInvertedIndex()
        with open(filename, 'rb') as f:
            while True:
                # Read term length
                term_length_bytes = f.read(2)
                if not term_length_bytes:
                    break

                term_length = struct.unpack("H", term_length_bytes)[0]
                term = f.read(term_length).decode('utf-8')

                # Read number of chunks
                num_chunks = struct.unpack("I", f.read(4))[0]

                # Read boundaries
                boundaries = []
                for _ in range(num_chunks):
                    start_id, end_id = struct.unpack("II", f.read(8))
                    boundaries.append((start_id, end_id))

                # Read chunks
                chunks = []
                for _ in range(num_chunks):
                    chunk_length = struct.unpack("I", f.read(4))[0]
                    chunk = f.read(chunk_length)
                    chunks.append(chunk)

                index._compressed_index[term] = ChunkedPostings(chunks=chunks, boundaries=boundaries)

        return index

    def get_posting_list(self, term: str) -> Optional[PostingList]:
        """
        Returns a PostingList object for the given term.

        :param term: The term to fetch the PostingList for
        :return: A PostingList object or None if term not found
        """
        chunked_postings = self._compressed_index.get(term)
        if not chunked_postings:
            return None

        # Generate skip pointers (every N chunks)
        skip_pointers = []
        for start_id, _ in chunked_postings.boundaries[::2]:  # Skip every other chunk boundary
            skip_pointers.append(start_id)

        return PostingList(
            compressed_chunks=chunked_postings.chunks,
            chunk_boundaries=chunked_postings.boundaries,
            skip_pointers=skip_pointers
        )

    def compress_and_add_postings(self, term: str, doc_ids: List[int], frequencies: List[int]) -> None:
        """
        Compress and add postings for a term using chunked compression.

        :param term: The term for which the postings are being added
        :param doc_ids: List of document IDs
        :param frequencies: List of term frequencies
        """
        chunked_postings = self._create_posting_chunks(doc_ids, frequencies)

        if term in self._compressed_index:
            # Merge with existing postings
            existing = self._compressed_index[term]

            # Append new chunks and boundaries
            existing.chunks.extend(chunked_postings.chunks)
            existing.boundaries.extend(chunked_postings.boundaries)

            # Sort chunks by doc_id boundaries if needed
            if existing.boundaries[-2][0] > chunked_postings.boundaries[0][0]:
                # Create pairs of (boundary, chunk) and sort them
                pairs = list(zip(existing.boundaries, existing.chunks))
                pairs.sort(key=lambda x: x[0][0])  # Sort by start_doc_id

                # Unzip the sorted pairs
                existing.boundaries, existing.chunks = zip(*pairs)
        else:
            self._compressed_index[term] = chunked_postings

    def get_terms(self):
        """Returns all terms in the index."""
        return self._compressed_index.keys()

    def get_uncompressed_postings(self, term: str) -> List[Posting]:
        """
        Gets all postings for a term (uncompressed).
        Warning: This decompresses all chunks - use get_posting_list() for better performance.

        :param term: The term to fetch postings for
        :return: List of Posting objects
        """
        chunked_postings = self._compressed_index.get(term)
        if not chunked_postings:
            return []

        all_doc_ids = []
        all_frequencies = []

        for chunk in chunked_postings.chunks:
            doc_ids, frequencies = CompressionTools.p_for_delta_decompress(chunk)
            all_doc_ids.extend(doc_ids)
            all_frequencies.extend(frequencies)

        return [Posting(doc_id=doc_id, payload=freq)
                for doc_id, freq in zip(all_doc_ids, all_frequencies)]