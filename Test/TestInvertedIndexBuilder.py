import unittest
import time
import os
from InvertedIndex.InvertedIndexBuilder import InvertedIndexBuilder
from InvertedIndex.Merger import Merger
from Utils.CollectionLoader import CollectionLoader
from Utils.Preprocessing import Preprocessing

class TestInvertedIndexBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures for CollectionLoader and Preprocessing."""
        cls.collection_loader = CollectionLoader()
        cls.preprocessing = Preprocessing()
        cls.merger = Merger()

    def setUp(self):
        """Create a fresh InvertedIndexBuilder instance for each test."""
        self.index_builder = InvertedIndexBuilder(
            collection_loader=self.collection_loader,
            preprocessing=self.preprocessing,
            merger=self.merger
        )

    def tearDown(self):
        """Ensure any intermediate index files are cleaned up after each test."""
        # Assuming partial indices are written to disk and need cleanup
        for i in range(1, 9):  # Assuming 8 partial indices are created for testing
            index_path = f'Compressed_Index_{i}.vb'
            if os.path.exists(index_path):
                os.remove(index_path)
                print(f"Deleted intermediate index file: {index_path}")

    def test_build_full_index(self):
        """Test the complete index building and merging process with PForDelta compression."""
        start_time = time.time()

        try:
            # Build the full index
            self.index_builder.build_full_index()
            build_time = time.time() - start_time

            # Retrieve the final inverted index
            index = self.index_builder.get_index()
            terms = index.get_terms()

            # Basic validation to ensure the index has terms
            self.assertGreater(len(terms), 0, "Index should contain terms")

            # Select a sample of terms to validate the index content
            sample_terms = terms[:5]  # First 5 terms for testing

            for term in sample_terms:
                postings = index.get_uncompressed_postings(term)

                # Check if postings list exists and is non-empty
                self.assertIsNotNone(postings, f"Postings should exist for term '{term}'")
                self.assertGreater(len(postings), 0, f"Postings list should not be empty for term '{term}'")

                # Validate the structure of each posting
                first_posting = postings[0]
                self.assertIsInstance(first_posting.doc_id, int, "Document ID should be an integer")

                # Ensure postings are correctly merged without duplicates
                doc_ids = [p.doc_id for p in postings]
                unique_doc_ids = set(doc_ids)
                self.assertEqual(len(doc_ids), len(unique_doc_ids),
                                 f"Duplicate document IDs found in postings for term '{term}'")

            # Output performance metrics
            print(f"\nIndex Building Performance:")
            print(f"- Build time: {build_time:.2f} seconds")
            print(f"- Total unique terms: {len(terms)}")
            print(f"- Sample term frequencies:")
            for term in sample_terms:
                print(f"  - '{term}': {len(index.get_compressed_postings(term))} documents")

        except Exception as e:
            self.fail(f"Index building failed with error: {str(e)}")

    def test_build_partial_index(self):
        """Test partial index building with sampled documents"""
        start_time = time.time()
        expected_sample_size = 10000  # The sample size we expect from build_partial_index

        try:
            # Build the partial index
            index = self.index_builder.build_partial_index()
            build_time = time.time() - start_time

            terms = index.get_terms()

            # Basic validation
            self.assertGreater(len(terms), 0, "Partial index should contain terms")

            # Validate document count
            all_doc_ids = set()
            for term in terms:
                postings = index.get_postings(term)
                doc_ids = {posting.doc_id for posting in postings}
                all_doc_ids.update(doc_ids)

            # Check if number of unique documents is less than or equal to sample size
            self.assertLessEqual(
                len(all_doc_ids),
                expected_sample_size,
                f"Number of unique documents ({len(all_doc_ids)}) exceeds expected sample size ({expected_sample_size})"
            )

            # Select sample terms for detailed validation
            sample_terms = terms[:5] if len(terms) >= 5 else terms  # First 5 terms or all if less

            for term in sample_terms:
                postings = index.get_postings(term)

                # Verify postings exist
                self.assertIsNotNone(postings, f"Postings should exist for term '{term}'")
                self.assertGreater(len(postings), 0, f"Postings list should not be empty for term '{term}'")

                # Verify posting structure
                first_posting = postings[0]
                self.assertIsInstance(first_posting.doc_id, int, "Document ID should be an integer")

                # Verify document IDs are within valid range
                for posting in postings:
                    self.assertGreater(posting.doc_id, 0, "Document IDs should be positive integers")

                # Verify postings are unique for each term
                doc_ids = [p.doc_id for p in postings]
                unique_doc_ids = set(doc_ids)

                # Print debugging information if duplicates are found
                if len(doc_ids) != len(unique_doc_ids):
                    duplicate_ids = [did for did in doc_ids if doc_ids.count(did) > 1]
                    print(f"\nDuplicate document IDs found for term '{term}':")
                    print(f"- All doc IDs: {doc_ids}")
                    print(f"- Duplicate IDs: {duplicate_ids}")

                self.assertEqual(
                    len(doc_ids),
                    len(unique_doc_ids),
                    f"Duplicate document IDs found in postings for term '{term}'. "
                    f"Total postings: {len(doc_ids)}, Unique postings: {len(unique_doc_ids)}"
                )

            # Print performance metrics and index statistics
            print(f"\nPartial Index Building Performance:")
            print(f"- Build time: {build_time:.2f} seconds")
            print(f"- Total unique terms: {len(terms)}")
            print(f"- Total unique documents: {len(all_doc_ids)}")
            print(f"- Average postings per term: {sum(len(index.get_postings(t)) for t in terms) / len(terms):.2f}")
            print(f"- Sample term frequencies:")
            for term in sample_terms:
                print(f"  - '{term}': {len(index.get_postings(term))} documents")

        except Exception as e:
            self.fail(f"Partial index building failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main()
