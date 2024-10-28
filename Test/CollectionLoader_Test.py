import unittest
import pandas as pd
from Utils.CollectionLoader import CollectionLoader  # Ensure this matches the actual module name

class CollectionLoader_Test(unittest.TestCase):
    def setUp(self):
        self.file_path = 'C:\\Users\pietr\OneDrive\Documenti\GitHub\MIRCV-Project\Files\collection.tsv'
        self.processor = CollectionLoader(self.file_path, chunk_size=100000)

    def test_process_chunks(self):
        df = self.processor.process_chunks()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        print("Some lines:")
        print(df.shape)

    def test_sample_lines(self):
        sampled_df = self.processor.sample_lines(num_lines=10)

        # Debug prints
        print("Sampled DataFrame:")
        print(sampled_df)
        print("Number of sampled rows:", len(sampled_df))

        # Check the number of sampled rows
        self.assertEqual(len(sampled_df), 10)

        # Check the columns
        expected_columns = set(pd.read_csv(self.file_path, sep='\t', nrows=1).columns)
        print("Expected columns:", expected_columns)
        print("Sampled DataFrame columns:", set(sampled_df.columns))
        self.assertTrue(set(sampled_df.columns) == expected_columns)

if __name__ == '__main__':
    unittest.main()
