import unittest
import time
from InvertedIndex.InvertedIndexBuilder import InvertedIndexBuilder
from Utils.CollectionLoader import CollectionLoader
from Utils.Preprocessing import Preprocessing

class TestInvertedIndexBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures for CollectionLoader and Preprocessing."""
        cls.collection_loader = CollectionLoader()
        cls.preprocessing = Preprocessing()

    def setUp(self):
        """Create a fresh InvertedIndexBuilder instance for each test."""
        self.index_builder = InvertedIndexBuilder(
            collection_loader=self.collection_loader,
            preprocessing=self.preprocessing,
            chunk_size=500000
        )

    def test_build_full_index(self):
        """Test complete index building and merging process"""
        start_time = time.time()

        try:
            # Build the index
            self.index_builder.build_full_index()
            build_time = time.time() - start_time

            # Get the final index
            index = self.index_builder.get_index()
            terms = index.get_terms()

            # Basic validation
            self.assertGreater(len(terms), 0, "Index should contain terms")

            # Select sample terms for detailed validation
            sample_terms = terms[:5]  # First 5 terms for testing

            for term in sample_terms:
                postings = index.get_postings(term)

                # Verify postings exist
                self.assertIsNotNone(postings, f"Postings should exist for term '{term}'")
                self.assertGreater(len(postings), 0, f"Postings list should not be empty for term '{term}'")

                # Verify posting structure
                first_posting = postings[0]
                self.assertIsInstance(first_posting.doc_id, int, "Document ID should be an integer")

                # Verify postings are properly merged (no duplicates)
                doc_ids = [p.doc_id for p in postings]
                unique_doc_ids = set(doc_ids)
                self.assertEqual(len(doc_ids), len(unique_doc_ids),
                                 f"Duplicate document IDs found in postings for term '{term}'")

            # Print performance metrics
            print(f"\nIndex Building Performance:")
            print(f"- Build time: {build_time:.2f} seconds")
            print(f"- Total unique terms: {len(terms)}")
            print(f"- Sample term frequencies:")
            for term in sample_terms:
                print(f"  - '{term}': {len(index.get_postings(term))} documents")

        except Exception as e:
            self.fail(f"Index building failed with error: {str(e)}")


if __name__ == '__main__':
    unittest.main()