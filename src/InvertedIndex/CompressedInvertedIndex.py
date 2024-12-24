import struct
from typing import List

from InvertedIndex.Posting import Posting
from Utils.CompressionTools import CompressionTools


class CompressedInvertedIndex:
    def __init__(self):
        # Dict
        self._compressed_index = {}

    def write_compressed_index_to_file(self, filename: str) -> None:
        """
        Writes the compressed inverted index to a file for persistent storage.

        Args:
            filename (str): the path of the final file.
            """
        with open(filename, 'wb') as f:
            for term, compressed_data in self._compressed_index.items():
                # Write the term as a UTF-8 encoded string
                term_bytes = term.encode('utf-8')
                f.write(struct.pack("H", len(term_bytes)))  # Write term length (2 bytes)
                f.write(term_bytes)  # Write term

                # Write the length of the compressed data (4 bytes)
                f.write(struct.pack("I", len(compressed_data)))
                f.write(compressed_data)  # Write compressed doc_ids and frequencies

    @staticmethod
    def load_compressed_index_to_memory(filepath: str) -> 'CompressedInvertedIndex':
        """
        Loads a compressed inverted index into memory in compressed form.

        Args:
            filepath (str): The path of the index to load.

        Returns:
            CompressedInvertedIndex: The compressed inverted index structure saved in the file.
        """
        index = CompressedInvertedIndex()
        with open(filepath, 'rb') as f:
            while True:
                # Read the term length (2 bytes)
                term_length_bytes = f.read(2)
                if not term_length_bytes:
                    break  # End of file

                term_length = struct.unpack("H", term_length_bytes)[0]
                term = f.read(term_length).decode('utf-8')  # Decode the term

                # Read the compressed data length (4 bytes)
                compressed_length_bytes = f.read(4)
                if not compressed_length_bytes:
                    raise ValueError("Unexpected end of file when reading compressed length")

                compressed_length = struct.unpack("I", compressed_length_bytes)[0]
                compressed_data = f.read(compressed_length)

                # Integrity check
                if len(compressed_data) != compressed_length:
                    raise ValueError("Mismatch between expected and actual compressed length")

                # Store the compressed data in memory
                index._compressed_index[term] = compressed_data

        return index

    def get_compressed_postings(self, term: str) -> bytes:
        """
        Fetches the compressed postings for a given term.

        Args:
            term (str): The term for which the postings are being fetched.

        Returns:
            bytes: The compressed postings.
        """
        return self._compressed_index.get(term, b'')  # Return empty bytes if term not found

    def add_compressed_postings(self, term: str, compressed_postings: bytes) -> None:
        """
        Add compressed postings for a term to the index.

        Args:
            term: The term for which the postings are being added.
            compressed_postings: The compressed postings as a byte string.
        """
        if term in self._compressed_index:
            # If the term already exists, concatenate the new postings
            self._compressed_index[term] += compressed_postings
        else:
            # Otherwise, create a new entry for the term
            self._compressed_index[term] = compressed_postings

    def get_terms(self):
        """
        Getter for terms.
        """
        return self._compressed_index.keys()

    def get_uncompressed_postings(self, term: str) -> List[Posting]:
        """
        Fetches the uncompressed postings for a given term as a list of Posting objects.

        Args:
            term (str): The term for which the postings are being fetched.

        Returns:
            List[Posting]: A list of Posting objects, or an empty list if the term is not found.
        """
        compressed_postings = self.get_compressed_postings(term)
        if compressed_postings:
            doc_ids, frequencies = CompressionTools.p_for_delta_decompress(compressed_postings)
            # Convert doc_ids and frequencies to a list of Posting objects
            list_postings = [Posting(doc_id=doc_id, payload=freq) for doc_id, freq in zip(doc_ids, frequencies)]
            return list_postings
        return []

    def compress_and_add_postings(self, term: str, doc_ids: List[int], frequencies: List[int]) -> None:
        """
        Compress and add postings for a term. Useful for testing of other methods.

        Args:
            term (str): The term for which the postings are being added.
            doc_ids (List[int]): A list of document IDs.
            frequencies (List[int]): A list of term frequencies corresponding to the doc IDs.
        """
        # Compress doc_ids and frequencies together
        compressed_data = CompressionTools.p_for_delta_compress(doc_ids, frequencies)
        self.add_compressed_postings(term, compressed_data)
