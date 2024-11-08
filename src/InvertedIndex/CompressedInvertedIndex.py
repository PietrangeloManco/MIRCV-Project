import struct

# Inverted Index structure class
class CompressedInvertedIndex:
    def __init__(self):
        self._compressed_index = {}

    def write_compressed_index_to_file(self, filename: str) -> None:
        """Writes the inverted index to a file using PForDelta compression if needed."""
        with open(filename, 'wb') as f:
            for term, compressed_doc_ids in self._compressed_index.items():
                # Write the term as a UTF-8 encoded string
                term_bytes = term.encode('utf-8')
                f.write(struct.pack("H", len(term_bytes)))  # Write term length (2 bytes)
                f.write(term_bytes)  # Write term

                # Write the length of the compressed doc_ids (4 bytes)
                f.write(struct.pack("I", len(compressed_doc_ids)))
                f.write(compressed_doc_ids)  # Write compressed doc_ids

                # Optionally, handle payloads if needed

    @staticmethod
    def load_compressed_index_to_memory(filename: str) -> 'CompressedInvertedIndex':
        """Loads a compressed inverted index into memory (in compressed form)."""
        index = CompressedInvertedIndex()
        with open(filename, 'rb') as f:
            while True:
                # Read the term length (2 bytes)
                term_length_bytes = f.read(2)
                if not term_length_bytes:
                    break  # End of file

                term_length = struct.unpack("H", term_length_bytes)[0]
                term = f.read(term_length).decode('utf-8')  # Decode the term

                # Read the compressed doc_ids length (4 bytes)
                compressed_length_bytes = f.read(4)
                if not compressed_length_bytes:
                    raise ValueError("Unexpected end of file when reading compressed length")

                compressed_length = struct.unpack("I", compressed_length_bytes)[0]
                compressed_doc_ids = f.read(compressed_length)

                if len(compressed_doc_ids) != compressed_length:
                    raise ValueError("Mismatch between expected and actual compressed length")

                # Store the compressed data in memory
                index._compressed_index[term] = compressed_doc_ids  # Use assignment, not append

        return index

    def get_compressed_postings(self, term: str) -> bytes:
        """Fetches the compressed postings for a given term."""
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
        return self._compressed_index.keys()