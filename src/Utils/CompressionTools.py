import struct
from typing import List, Tuple


class CompressionTools:

    @staticmethod
    def p_for_delta_decompress(data: bytes) -> Tuple[List[int], List[int]]:
        """
        Decompresses data into a list of doc IDs and term frequencies using the p for delta compression algorithm.

        Args:
            data(bytes): Data to decompress (postings).

        Returns:
            Tuple[List[int], List[int]]: The list of doc_ids and relative frequencies.
        """
        if len(data) == 0:
            return [], []  # Handle empty data gracefully

        bit_width = struct.unpack("B", data[:1])[0]
        data = data[1:]

        deltas = []
        frequencies = []

        # Calculate the number of elements for deltas and frequencies
        total_length = len(data)
        half_length = total_length // 2

        # Split data into deltas and frequencies
        delta_data = data[:half_length]
        freq_data = data[half_length:]

        # Decompress deltas
        for i in range(0, len(delta_data), bit_width):
            delta = int.from_bytes(delta_data[i:i + bit_width], byteorder='big')
            deltas.append(delta)

        # Decompress frequencies
        for i in range(0, len(freq_data), bit_width):
            freq = int.from_bytes(freq_data[i:i + bit_width], byteorder='big')
            frequencies.append(freq)

        # Ensure doc_ids and frequencies are aligned
        if len(deltas) != len(frequencies):
            raise ValueError("Mismatched number of deltas and frequencies.")

        # Reconstruct original doc IDs from deltas
        doc_ids = [deltas[0]]
        for delta in deltas[1:]:
            doc_ids.append(doc_ids[-1] + delta)

        return doc_ids, frequencies

    @staticmethod
    def p_for_delta_compress(doc_ids: List[int], frequencies: List[int]) -> bytes:
        """
        Compresses data into a list of doc IDs and term frequencies using the p for delta compression algorithm.

        Args:
            doc_ids(List[int]): Doc_ids to compress.
            frequencies(List[int]): Relative term frequencies to compress.

        Returns:
            bytes: Compressed list of doc_ids and frequencies.
        """
        if len(doc_ids) != len(frequencies):
            raise ValueError("doc_ids and frequencies lists must be of the same length.")

        if len(doc_ids) == 0:  # Handle empty input lists
            return b""

        # Delta encode the doc IDs
        deltas = [doc_ids[0]] + [doc_ids[i] - doc_ids[i - 1] for i in range(1, len(doc_ids))]

        # Determine bit width
        max_bits = max((delta.bit_length() for delta in deltas), default=1)
        max_bits_freq = max((freq.bit_length() for freq in frequencies), default=1)
        bit_width = (max(max_bits, max_bits_freq) + 7) // 8  # Convert bits to bytes

        # Compress both deltas and frequencies
        compressed_bytes = bytearray()
        compressed_bytes.extend(struct.pack("B", bit_width))

        # Compress deltas
        for delta in deltas:
            compressed_bytes.extend(delta.to_bytes(bit_width, byteorder='big'))

        # Compress frequencies
        for freq in frequencies:
            compressed_bytes.extend(freq.to_bytes(bit_width, byteorder='big'))

        return bytes(compressed_bytes)
