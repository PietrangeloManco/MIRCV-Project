import struct
from collections import defaultdict
from typing import Any, List, Iterator
from src.InvertedIndex.Posting import Posting

# Inverted Index structure class
class InvertedIndex:

    def __init__(self):
        self._index = defaultdict(list)

    # Add a posting to the inverted index
    def add_posting(self, term: str, doc_id: int, payload: Any = None) -> None:
        """Adds a document to the posting list of a term."""
        self._index[term].append(Posting(doc_id, payload))

    def add_postings(self, term: str, postings: Iterator[Posting]) -> None:
        """Adds a collection of postings for a given term."""
        self._index[term].extend(postings)

    # Get the posting for a given term from the inverted index
    def get_postings(self, term: str) -> List[Posting]:
        """Fetches the posting list for a given term."""
        return self._index.get(term)

    # Get the list of unique terms in the inverted index
    def get_terms(self) -> List[str]:
        """Returns all unique terms in the index."""
        return list(self._index.keys())

    # Write the inverted index to a file
    def write_to_file(self, filename_index: str) -> None:
        """Saves the index to a textfile."""
        with open(filename_index, "w") as f:
            for term, postings in self._index.items():
                f.write(term)
                for posting in postings:
                    f.write(f" {posting.doc_id}")
                    if posting.payload:
                        f.write(f":{str(posting.payload)}")
                f.write("\n")

    @staticmethod
    def load_from_file(filename_index: str) -> 'InvertedIndex':
        """Loads an inverted index from a textfile."""
        index = InvertedIndex()
        with open(filename_index, "r") as f:
            for line in f:
                terms = line.strip().split()
                term = terms[0]
                for term_data in terms[1:]:
                    doc_id, *payload = term_data.split(":")
                    payload = ":".join(payload) if payload else None
                    index.add_posting(term, int(doc_id), payload)
        return index

    @staticmethod
    def read_index_terms(filename: str) -> set:
        """Read unique terms from the index file."""
        terms = set()
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    terms.add(parts[0])  # Add the term (first part of the line)
        return terms

    @staticmethod
    def read_index_postings(filename: str, batch_terms: set) -> dict:
        """Read postings for specified terms from the index file."""
        postings_dict = {}
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                term = parts[0]
                if term in batch_terms:
                    postings = []
                    for posting_data in parts[1:]:
                        doc_id, *payload = posting_data.split(":")
                        payload = ":".join(payload) if payload else None
                        postings.append(Posting(int(doc_id), payload))
                    postings_dict[term] = postings
        return postings_dict

        # Helper function for variable byte encoding

    @staticmethod
    def pfor_delta_compress(doc_ids: List[int]) -> bytes:
        # Step 1: Apply Delta Encoding
        deltas = [doc_ids[0]] + [doc_ids[i] - doc_ids[i - 1] for i in range(1, len(doc_ids))]

        # Step 2: Compress using PForDelta
        # (Simplified implementation: using fixed bit-width for simplicity)
        compressed_bytes = bytearray()
        max_bits = max((delta.bit_length() for delta in deltas), default=1)
        bit_width = (max_bits + 7) // 8  # Convert bits to bytes

        # Write the bit width
        compressed_bytes.extend(struct.pack("B", bit_width))

        # Write all deltas using the bit width
        for delta in deltas:
            compressed_bytes.extend(delta.to_bytes(bit_width, byteorder='big'))

        return bytes(compressed_bytes)

    @staticmethod
    def pfor_delta_decompress(data: bytes) -> List[int]:
        # Read the bit width
        bit_width = struct.unpack("B", data[:1])[0]
        deltas = []
        data = data[1:]

        # Read deltas
        for i in range(0, len(data), bit_width):
            delta = int.from_bytes(data[i:i + bit_width], byteorder='big')
            deltas.append(delta)

        # Convert deltas back to original doc_ids
        doc_ids = [deltas[0]]
        for delta in deltas[1:]:
            doc_ids.append(doc_ids[-1] + delta)

        return doc_ids

    def write_compressed_index_to_file(self, filename: str, is_compressed: bool = False) -> None:
        """Writes the inverted index to a file using PForDelta compression if needed."""
        with open(filename, 'wb') as f:
            for term, postings in self._index.items():
                # Write the term as a UTF-8 encoded string
                term_bytes = term.encode('utf-8')
                f.write(struct.pack("H", len(term_bytes)))
                f.write(term_bytes)

                # Check if the doc_ids are already compressed
                if is_compressed:
                    compressed_doc_ids = self.get_compressed_postings(term)
                else:
                    doc_ids = [posting.doc_id for posting in postings]
                    compressed_doc_ids = self.pfor_delta_compress(doc_ids)

                # Write the compressed doc_ids
                f.write(struct.pack("I", len(compressed_doc_ids)))
                f.write(b''.join(compressed_doc_ids))

                # Optionally, handle payloads (not compressed in this example)
                # Extend this as needed for handling payloads

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
                doc_ids = index.pfor_delta_decompress(compressed_doc_ids)

                # Add the postings to the index
                for doc_id in doc_ids:
                    index.add_posting(term, doc_id)

        return index

    @staticmethod
    def load_compressed_index_to_memory(filename: str) -> 'InvertedIndex':
        """Loads a compressed inverted index into memory (in compressed form)."""
        index = InvertedIndex()
        with open(filename, 'rb') as f:
            while True:
                # Read the term length (2 bytes)
                term_length_bytes = f.read(2)
                if not term_length_bytes:
                    break  # End of file

                term_length = struct.unpack("H", term_length_bytes)[0]
                term = f.read(term_length).decode('utf-8')  # Decode the term correctly

                # Read the compressed doc_ids length (4 bytes)
                compressed_length_bytes = f.read(4)
                if not compressed_length_bytes:
                    raise ValueError("Unexpected end of file when reading compressed length")

                compressed_length = struct.unpack("I", compressed_length_bytes)[0]
                compressed_doc_ids = f.read(compressed_length)

                if len(compressed_doc_ids) != compressed_length:
                    raise ValueError("Mismatch between expected and actual compressed length")

                # Store the compressed data in memory
                index._index[term].append(compressed_doc_ids)

        return index

    def get_compressed_postings(self, term: str) -> List[bytes]:
        """Fetches the compressed postings for a given term."""
        return self._index.get(term, [])

    def decompress_postings(self, term: str) -> List[int]:
        """Decompress the postings for a given term and return the document IDs."""
        compressed_postings = self.get_compressed_postings(term)
        doc_ids = []
        for compressed_data in compressed_postings:
            doc_ids.extend(self.pfor_delta_decompress(compressed_data))
        return doc_ids

    def add_compressed_postings(self, term: str, compressed_data: list[bytes]) -> None:
        """Adds a compressed posting (in bytes) to the inverted index for a given term."""
        self._index[term].extend(compressed_data)