import unittest

from src.Utils.CompressionTools import CompressionTools


class TestCompressionTools(unittest.TestCase):

    def test_p_for_delta_compress_decompress(self):
        """Test that data compressed and then decompressed match the original doc_ids and frequencies"""
        test_cases = [
            ([1, 5, 10], [2, 3, 1]),  # Normal case with small numbers
            ([1000, 2000, 3000], [1, 1, 1]),  # Larger numbers with constant frequency
            ([1], [1]),  # Single element edge case
            ([10, 20, 30, 40], [5, 6, 7, 8])  # Case with increasing numbers
        ]

        for doc_ids, frequencies in test_cases:
            with self.subTest(doc_ids=doc_ids, frequencies=frequencies):
                # Compress the doc_ids and frequencies
                compressed_data = CompressionTools.p_for_delta_compress(doc_ids, frequencies)

                # Decompress and check if we get the original data back
                decompressed_doc_ids, decompressed_frequencies = CompressionTools.p_for_delta_decompress(
                    compressed_data)

                # Assert the decompressed values match the original input
                self.assertEqual(doc_ids, decompressed_doc_ids)
                self.assertEqual(frequencies, decompressed_frequencies)

    def test_mismatched_doc_ids_and_frequencies(self):
        """Test that ValueError is raised when doc_ids and frequencies lists have different lengths"""
        with self.assertRaises(ValueError):
            CompressionTools.p_for_delta_compress([1, 5], [2])  # Mismatched lengths

    def test_empty_input(self):
        """Test handling of empty input lists"""
        # Empty input should still compress and decompress correctly
        doc_ids = []
        frequencies = []

        compressed_data = CompressionTools.p_for_delta_compress(doc_ids, frequencies)
        decompressed_doc_ids, decompressed_frequencies = CompressionTools.p_for_delta_decompress(compressed_data)

        self.assertEqual(doc_ids, decompressed_doc_ids)
        self.assertEqual(frequencies, decompressed_frequencies)

    def test_single_element_input(self):
        """Test single element list"""
        doc_ids = [1]
        frequencies = [1]

        compressed_data = CompressionTools.p_for_delta_compress(doc_ids, frequencies)
        decompressed_doc_ids, decompressed_frequencies = CompressionTools.p_for_delta_decompress(compressed_data)

        self.assertEqual(doc_ids, decompressed_doc_ids)
        self.assertEqual(frequencies, decompressed_frequencies)

    def test_large_input(self):
        """Test larger input lists"""
        doc_ids = list(range(1, 1001))
        frequencies = [i % 10 + 1 for i in range(1000)]

        compressed_data = CompressionTools.p_for_delta_compress(doc_ids, frequencies)
        decompressed_doc_ids, decompressed_frequencies = CompressionTools.p_for_delta_decompress(compressed_data)

        self.assertEqual(doc_ids, decompressed_doc_ids)
        self.assertEqual(frequencies, decompressed_frequencies)

    def test_bit_width_calculation(self):
        """Test that bit width is calculated correctly based on the data"""
        doc_ids = [10, 1000, 100000]
        frequencies = [1, 2, 3]

        compressed_data = CompressionTools.p_for_delta_compress(doc_ids, frequencies)

        # Decompress and check the correct number of elements
        decompressed_doc_ids, decompressed_frequencies = CompressionTools.p_for_delta_decompress(compressed_data)

        self.assertEqual(doc_ids, decompressed_doc_ids)
        self.assertEqual(frequencies, decompressed_frequencies)


if __name__ == "__main__":
    unittest.main()
