import unittest
from src.InvertedIndex.InvertedIndex import InvertedIndex
from src.InvertedIndex.InvertedIndexBuilder import InvertedIndexBuilder
from src.Utils.CollectionLoader import CollectionLoader
from src.Utils.Preprocessing import Preprocessing

# Testing class for the inverted index builder
class TestInvertedIndexBuilder(unittest.TestCase):

    def setUp(self):
        self.collection_loader = CollectionLoader()
        self.preprocessing = Preprocessing()
        self.index_builder = InvertedIndexBuilder(self.collection_loader, self.preprocessing)

    # Tests the full collection indexing method
    def test_build_full_index(self):
        # Build the index
        self.index_builder.build_full_index()
        # Get the built index
        index = self.index_builder.get_index()
        print(index.get_terms())

    # Tests the get index method
    def test_get_index(self):
        # Ensure get_index returns an instance of InvertedIndex
        self.assertIsInstance(self.index_builder.get_index(), InvertedIndex)

if __name__ == '__main__':
    unittest.main()
