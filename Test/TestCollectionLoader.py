import unittest
import pandas as pd
import numpy as np
from src.Utils.CollectionLoader import CollectionLoader


class TestCollectionLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collection_path = 'C:\\Users\\pietr\\OneDrive\\Documenti\\GitHub\\MIRCV-Project\\Files\\collection.tar.gz'
        cls.chunk_size = 1000  # Smaller chunk size for testing
        cls.loader = CollectionLoader(
            file_path=cls.collection_path,
            chunk_size=cls.chunk_size
        )

    def test_get_total_docs(self):
        """Test if total document count is returned correctly"""
        total_docs = self.loader.get_total_docs()
        self.assertGreater(total_docs, 0)
        print(f"Total documents in collection: {total_docs}")

    def test_process_chunks_iterator(self):
        """Test if chunk processing works correctly"""
        # Process first few chunks
        chunks_processed = 0
        total_rows = 0

        for chunk in self.loader.process_chunks():
            # Test chunk properties
            self.assertIsInstance(chunk, pd.DataFrame)
            self.assertEqual(list(chunk.columns), ['index', 'text'])

            # Verify data types - allow both int32 and int64
            self.assertTrue(
                chunk['index'].dtype in (np.int32, np.int64),
                f"Index dtype {chunk['index'].dtype} is not an integer type"
            )
            self.assertEqual(chunk['text'].dtype, 'object')

            # Verify chunk size
            self.assertLessEqual(len(chunk), self.chunk_size)

            # Verify data validity
            self.assertTrue(chunk['index'].notnull().all())
            self.assertTrue(chunk['text'].notnull().all())

            total_rows += len(chunk)
            chunks_processed += 1

            # Process a few chunks for testing
            if chunks_processed >= 3:
                break

        print(f"Processed {chunks_processed} chunks, total rows: {total_rows}")
        self.assertGreater(total_rows, 0)

    def test_chunk_size_respect(self):
        """Test if chunk size is respected"""
        chunk = next(self.loader.process_chunks())
        self.assertLessEqual(len(chunk), self.chunk_size)

    def test_sample_lines(self):
        """Test sampling functionality"""
        num_samples = 10
        sampled_df = self.loader.sample_lines(num_lines=num_samples)

        # Test basic properties
        self.assertIsInstance(sampled_df, pd.DataFrame)
        self.assertEqual(len(sampled_df), num_samples)
        self.assertEqual(set(sampled_df.columns), {'index', 'text'})

        # Verify data types - allow both int32 and int64
        self.assertTrue(
            sampled_df['index'].dtype in (np.int32, np.int64),
            f"Index dtype {sampled_df['index'].dtype} is not an integer type"
        )
        self.assertEqual(sampled_df['text'].dtype, 'object')

        # Verify data validity
        self.assertTrue(sampled_df['index'].notnull().all())
        self.assertTrue(sampled_df['text'].notnull().all())

        print("Sample data:")
        print(sampled_df.head())

    def test_malformed_data_handling(self):
        """Test handling of chunks with different column counts"""
        # Process a few chunks to check for errors
        chunks_processed = 0
        for chunk in self.loader.process_chunks():
            self.assertEqual(list(chunk.columns), ['index', 'text'])
            # Verify integer type without specifying exact type
            self.assertTrue(
                np.issubdtype(chunk['index'].dtype, np.integer),
                f"Index dtype {chunk['index'].dtype} is not an integer type"
            )
            chunks_processed += 1
            if chunks_processed >= 2:
                break

    def test_custom_chunk_size(self):
        """Test if custom chunk size is respected"""
        custom_chunk_size = 500
        chunk = next(self.loader.process_chunks(chunk_size=custom_chunk_size))
        self.assertLessEqual(len(chunk), custom_chunk_size)


if __name__ == '__main__':
    unittest.main()
