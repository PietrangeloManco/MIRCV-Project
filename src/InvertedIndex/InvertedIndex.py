import struct
from collections import defaultdict
from typing import Any, List, Iterator

from Utils.CompressionTools import CompressionTools
from src.InvertedIndex.Posting import Posting


# Inverted Index structure class
class InvertedIndex:
    def __init__(self):
        self._index = defaultdict(list)

    # Add a posting to the inverted index
    def add_posting(self, term: str, doc_id: int, payload: Any = None) -> None:
        """Adds a document to the posting list of a term, with optional term frequency as payload."""
        self._index[term].append(Posting(doc_id, payload))

    def add_postings(self, term: str, postings: Iterator[Posting]) -> None:
        """Adds a collection of postings for a given term."""
        self._index[term].extend(postings)

    def get_postings(self, term: str) -> List[Posting]:
        """Fetches the posting list for a given term."""
        return self._index.get(term, [])

    # Get the list of unique terms in the inverted index
    def get_terms(self) -> List[str]:
        """Returns all unique terms in the index."""
        return list(self._index.keys())

    # Write the inverted index to a file
    def write_to_file(self, filename_index: str) -> None:
        """Saves the index to a text file with term frequencies as payloads."""
        with open(filename_index, "w") as f:
            for term, postings in self._index.items():
                f.write(term)
                for posting in postings:
                    f.write(f" {posting.doc_id}")
                    if posting.payload is not None:
                        f.write(f":{str(posting.payload)}")  # Write the payload (term frequency)
                f.write("\n")

    @staticmethod
    def load_from_file(filename_index: str) -> 'InvertedIndex':
        """Loads an inverted index from a text file, handling term frequencies as payloads."""
        index = InvertedIndex()
        with open(filename_index, "r") as f:
            for line in f:
                terms = line.strip().split()
                term = terms[0]
                for term_data in terms[1:]:
                    doc_id, *payload = term_data.split(":")
                    payload = int(payload[0]) if payload else None
                    index.add_posting(term, int(doc_id), payload)
        return index

    @staticmethod
    def load_compressed_index_from_file(filename: str) -> 'InvertedIndex':
        """Loads the inverted index from a compressed file using PForDelta decompression."""
        index = InvertedIndex()
        with open(filename, 'rb') as f:
            while True:
                # Read the term
                term_length_bytes = f.read(2)
                if not term_length_bytes:
                    break
                term_length = struct.unpack("H", term_length_bytes)[0]
                term = f.read(term_length).decode('utf-8')

                # Read the compressed doc_ids
                compressed_length = struct.unpack("I", f.read(4))[0]
                compressed_doc_ids = f.read(compressed_length)
                doc_ids, frequencies = CompressionTools.p_for_delta_decompress(compressed_doc_ids)
                # Add the postings to the index (without payload for now)
                for doc_id, frequency in zip(doc_ids, frequencies):
                    index.add_posting(term, doc_id, frequency)

        return index

    def write_index_compressed_to_file(self, filename: str) -> None:
        """Writes the inverted index to a file using PForDelta compression if needed."""
        with open(filename, 'wb') as f:
            for term, postings in self._index.items():
                # Write the term as a UTF-8 encoded string
                term_bytes = term.encode('utf-8')
                f.write(struct.pack("H", len(term_bytes)))
                f.write(term_bytes)

                doc_ids = [posting.doc_id for posting in postings]
                frequencies = [posting.payload for posting in postings]
                compressed_doc_ids = CompressionTools.p_for_delta_compress(doc_ids, frequencies)

                # Write the compressed doc_ids
                f.write(struct.pack("I", len(compressed_doc_ids)))
                f.write(compressed_doc_ids)
