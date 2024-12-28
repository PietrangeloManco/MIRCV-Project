import os
import time
import unittest

from Index.DocumentTable.DocumentTable import DocumentTable
from Index.InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from Index.InvertedIndex.InvertedIndexBuilder import InvertedIndexBuilder
from Index.InvertedIndex.Merger import Merger
from Index.Lexicon.Lexicon import Lexicon
from Utils.CollectionLoader import CollectionLoader
from Utils.Preprocessing import Preprocessing
from Utils.config import RESOURCES_PATH


class TestInvertedIndexBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up class test structures."""
        cls.collection_loader = CollectionLoader()
        cls.preprocessing = Preprocessing()
        cls.merger = Merger()
        cls.document_table = DocumentTable()
        cls.lexicon = Lexicon()
        cls.resources_path = RESOURCES_PATH

    def setUp(self):
        """Create a fresh InvertedIndexBuilder instance for each test."""
        self.index_builder = InvertedIndexBuilder(
            collection_loader=self.collection_loader,
            preprocessing=self.preprocessing,
            merger=self.merger,
            document_table=self.document_table,
            lexicon=self.lexicon
        )

    def tearDown(self):
        """Clean up any leftover files from the tests."""
        # Path to the files to remove after tests
        partial_index_path = self.resources_path + "Compressed_Index_1.vb"
        partial_document_table_path = self.resources_path + "partial_document_table.txt"
        partial_lexicon_table_path = self.resources_path + "partial_lexicon.txt"

        # Check if the file exists and delete it
        if os.path.exists(partial_index_path):
            os.remove(partial_index_path)
            print(f"Deleted partial index file: {partial_index_path}")

        if os.path.exists(partial_document_table_path):
            os.remove(partial_document_table_path)
            print(f"Deleted partial document table file: {partial_document_table_path}")

        if os.path.exists(partial_lexicon_table_path):
            os.remove(partial_lexicon_table_path)
            print(f"Deleted partial lexicon file: {partial_lexicon_table_path}")

    def test_already_built_full_structures(self):
        """Test the structures previously built work as expected."""

        try:
            print("Step 1: Loading the total number of documents...")
            total_docs = self.collection_loader.get_total_docs()
            print(f"Total documents loaded: {total_docs}")

            print("Step 2: Loading lexicon and document table into memory...")
            lexicon = Lexicon.load_from_file(self.index_builder.resources_path + "Lexicon")
            print("Lexicon loaded.")

            document_table = DocumentTable.load_from_file(self.index_builder.resources_path + "DocumentTable")
            print("Document table loaded.")

            # Initialize the inverted index
            index = CompressedInvertedIndex.load_compressed_index_to_memory(
                self.index_builder.resources_path + "InvertedIndex")
            print("Inverted Index loaded.")

            print("Step 3: Getting terms from the lexicon...")
            terms = lexicon.get_all_terms()
            print(f"Number of terms retrieved: {len(terms)}")
            self.assertGreater(len(terms), 0, "Full index should contain terms")

            # Sample terms early to limit the number of postings processed
            sample_terms = list(terms)[:100] if len(terms) >= 100 else terms
            print(f"Number of sample terms selected for detailed validation: {len(sample_terms)}")

            all_doc_ids = set()

            # Validate sample terms
            print("Step 4: Validating sample terms...")
            for i, term in enumerate(sample_terms):
                if i % 10 == 0:
                    print(f"Validating term {i + 1}/{len(sample_terms)}: '{term}'")

                # Get term metadata from the lexicon
                term_info = lexicon.get_term_info(term)
                if not term_info:
                    self.fail(f"Term '{term}' not found in the lexicon.")

                postings = index.get_uncompressed_postings(term)
                self.assertIsNotNone(postings, f"Postings should exist for term '{term}'")
                self.assertGreater(len(postings), 0, f"Postings list should not be empty for term '{term}'")

                # Validate postings
                first_posting = postings[0]
                self.assertIsInstance(first_posting.doc_id, int, "Document ID should be an integer")

                doc_ids = [p.doc_id for p in postings]
                unique_doc_ids = set(doc_ids)
                all_doc_ids.update(unique_doc_ids)

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

            # Verify document table using collected document IDs
            print("Step 5: Verifying document table...")
            document_table_ids = set(document_table._document_table.keys())
            print("Document table IDs loaded.")

            for doc_id in all_doc_ids:
                self.assertIn(doc_id, document_table_ids, f"Document table should contain the document ID {doc_id}")

            # Print statistics
            print("Step 6: Printing statistics...")
            print(f"- Total unique sample terms validated: {len(sample_terms)}")
            print(f"- Total unique documents found: {len(all_doc_ids)}")

        except Exception as e:
            self.fail(f"Full structures validation error: {str(e)}")

    def test_build_partial_index(self):
        """Test partial index building with randomly sampled documents."""
        start_time = time.time()
        expected_sample_size = 10000  # The sample size expected from build_partial_index (default)

        try:
            # Build the partial index
            self.index_builder.build_partial_index()
            index = self.index_builder.get_index()
            lexicon = self.index_builder.get_lexicon()
            document_table = self.index_builder.get_document_table()
            build_time = time.time() - start_time

            terms = index.get_terms()

            # Basic validation
            self.assertGreater(len(terms), 0, "Partial index should contain terms")

            # Validate document count
            all_doc_ids = set()
            for term in terms:
                postings = index.get_uncompressed_postings(term)
                doc_ids = {posting.doc_id for posting in postings}
                all_doc_ids.update(doc_ids)

            # Check if number of unique documents is less than or equal to sample size
            self.assertLessEqual(
                len(all_doc_ids),
                expected_sample_size,
                f"Number of unique documents ({len(all_doc_ids)}) exceeds expected sample size ({expected_sample_size})"
            )

            # Select sample terms for detailed validation
            sample_terms = list(terms)[:5] if len(terms) >= 5 else terms  # First 5 terms or all if less

            for term in sample_terms:
                postings = index.get_uncompressed_postings(term)

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

            # Verify lexicon contains expected terms
            for term in sample_terms:
                self.assertIn(term, lexicon.get_all_terms(), f"Lexicon should contain the term '{term}'")

            # Verify document table contains expected documents
            for doc_id in all_doc_ids:
                self.assertIn(doc_id, document_table._document_table,
                              f"Document table should contain the document ID {doc_id}")

            # Print performance metrics and index statistics
            print(f"\nPartial Index Building Performance:")
            print(f"- Build time: {build_time:.2f} seconds")
            print(f"- Total unique terms: {len(terms)}")
            print(f"- Total unique documents: {len(all_doc_ids)}")
            print(
                f"- Average postings per term: {sum(len(index.get_uncompressed_postings(t)) for t in terms) / len(terms):.2f}")
            print(f"- Sample term frequencies:")
            for term in sample_terms:
                postings = index.get_uncompressed_postings(term)
                num_documents = len(postings)  # Number of documents for the term
                sample_frequencies = [posting.payload for posting in postings]  # Get frequencies from postings
                print(f"  - '{term}': {num_documents} documents, Frequencies: {sample_frequencies}")

        except Exception as e:
            self.fail(f"Partial index building failed with error: {str(e)}")


if __name__ == '__main__':
    unittest.main()
