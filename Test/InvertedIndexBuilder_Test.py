import unittest
from unittest.mock import MagicMock

import pandas as pd

from src.InvertedIndex.InvertedIndex import InvertedIndex
from src.InvertedIndex.InvertedIndexBuilder import InvertedIndexBuilder
from Utils.CollectionLoader import CollectionLoader
from Utils.Preprocessing import Preprocessing

class TestInvertedIndexBuilder(unittest.TestCase):

    def setUp(self):
        # Mock the CollectionLoader and Preprocessing
        self.mock_collection_loader = MagicMock(spec=CollectionLoader)
        self.mock_preprocessing = MagicMock(spec=Preprocessing)

        # Create an instance of InvertedIndexBuilder with mocks
        self.index_builder = InvertedIndexBuilder(self.mock_collection_loader, self.mock_preprocessing)

    def test_build_index(self):
        # Mock the data returned by process_chunks
        self.mock_collection_loader.process_chunks.return_value = pd.DataFrame({
            'text': ["This is a test document.", "Another test document."]
        })

        # Mock the preprocessing steps
        self.mock_preprocessing.preprocess.side_effect = [
            ["this", "test", "document"],
            ["another", "test", "document"]
        ]

        # Build the index
        self.index_builder.build_index()

        # Get the built index
        index = self.index_builder.get_index()

        # Check if the index contains the expected postings
        self.assertEqual(len(index.get_postings("test")), 2)
        self.assertEqual(len(index.get_postings("this")), 1)
        self.assertEqual(len(index.get_postings("another")), 1)
        self.assertEqual(len(index.get_postings("document")), 2)

    def test_get_index(self):
        # Ensure get_index returns an instance of InvertedIndex
        self.assertIsInstance(self.index_builder.get_index(), InvertedIndex)

if __name__ == '__main__':
    unittest.main()
