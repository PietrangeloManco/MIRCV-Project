import unittest
import pandas as pd
from src.Utils.CollectionLoader import CollectionLoader

# Testing class for the collection loader
class CollectionLoader_Test(unittest.TestCase):

    def setUp(self):
        self.processor = CollectionLoader()

    # Tests the full collection loader method
    def test_process_chunks(self):
        df = self.processor.process_chunks()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        print("Some lines:")
        print(df.shape)

    # Tests the partial collection loader method
    def test_sample_lines(self):
        sampled_df = self.processor.sample_lines(num_lines=10)
        print("Sampled DataFrame:")
        print(sampled_df)
        print("Number of sampled rows:", len(sampled_df))
        self.assertEqual(len(sampled_df), 10)
        expected_columns = set(pd.read_csv(self.processor.file_path, sep='\t', nrows=1).columns)
        print("Expected columns:", expected_columns)
        print("Sampled DataFrame columns:", set(sampled_df.columns))
        #self.assertTrue(set(sampled_df.columns) == expected_columns)

if __name__ == '__main__':
    unittest.main()
