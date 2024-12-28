import struct
from collections import defaultdict
from typing import Any, List

from Utils.CompressionTools import CompressionTools
from src.InvertedIndex.Posting import Posting


class InvertedIndex:
    def __init__(self):
        # Default Dict
        self._index = defaultdict(list)

    def add_posting(self, term: str, doc_id: int, payload: Any = None) -> None:
        """
        Adds a document to the posting list of a term, with optional term frequency as payload.

        Args:
            term (str): The term to add the posting to.
            doc_id (int): The id of the document to add to the list.
            payload (Any): The optional frequency of the term in the document.
        """
        self._index[term].append(Posting(doc_id, payload))

    def get_postings(self, term: str) -> List[Posting]:
        """
        Fetches the posting list for a given term. Useful for testing purposes.

        Args:
            term (str): The term to fetch the list of.

        Returns:
            List[Posting]: The list of postings.
        """
        return self._index.get(term, [])

    @staticmethod
    def load_compressed_index_from_file(filepath: str) -> 'InvertedIndex':
        """
        Loads the inverted index from a compressed file using PForDelta decompression.
        Useful to test the write function.

        Args:
            filepath(str): The path of the file to load.

        Returns:
            InvertedIndex: an uncompressed InvertedIndex object.
        """
        index = InvertedIndex()
        with open(filepath, 'rb') as f:
            while True:
                # Read the term
                term_length_bytes = f.read(2)
                if not term_length_bytes:
                    break
                term_length = struct.unpack("H", term_length_bytes)[0]
                term = f.read(term_length).decode('utf-8')

                # Read the compressed data
                compressed_length = struct.unpack("I", f.read(4))[0]
                compressed_doc_ids = f.read(compressed_length)
                doc_ids, frequencies = CompressionTools.p_for_delta_decompress(compressed_doc_ids)
                for doc_id, frequency in zip(doc_ids, frequencies):
                    index.add_posting(term, doc_id, frequency)

        return index

    def write_index_compressed_to_file(self, filepath: str) -> None:
        """
        Writes the inverted index to a file using PForDelta compression.

        Args:
            filepath(str): The path where to write the compressed index to.
        """
        with open(filepath, 'wb') as f:
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
