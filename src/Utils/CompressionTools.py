from typing import List
import struct

class CompressionTools:

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

