import gc
import os
import unittest
import time
from InvertedIndex.InvertedIndexBuilder import InvertedIndexBuilder
from Utils.CollectionLoader import CollectionLoader
from Utils.CompressionTools import CompressionTools
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
        )

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
                postings = index.get_postings(term)

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
                print(f"  - '{term}': {len(index.get_postings(term))} documents")

        except Exception as e:
            self.fail(f"Index building failed with error: {str(e)}")

    def test__merge_compressed_indices(self):
        """
        Test merging multiple partial compressed indices in a hierarchical and recursive manner.
        Merges 8 initial compressed indices pair by pair until a single final index is created.
        """

        def get_file_size_mb(path: str) -> float:
            """Helper to get file size in MB"""
            return os.path.getsize(path) / (1024 * 1024)

        def log_merge_operation(pair: list, level: str):
            """Helper to log merge operations with file sizes"""
            sizes = [get_file_size_mb(p) for p in pair]
            print(f'\n{level}: Merging files of size {sizes[0]:.1f}MB and {sizes[1]:.1f}MB')
            print(f'Files: {pair}')

        def validate_index_integrity(index_path: str) -> bool:
            """Helper to validate index integrity by attempting to load and decompress all postings"""
            try:
                print("Trying to load into memory")
                index = self.index_builder.inverted_index.load_compressed_index_to_memory(index_path)
                print("Trying to get terms")
                i = 0
                for term in index.get_terms():
                    postings = index.get_compressed_postings(term)
                    i += 1
                    for posting in postings:
                        # Attempt to decompress and ensure no errors are raised
                        if i % 100000 == 0:
                            print(f"Postings for term {term} are {CompressionTools.pfor_delta_decompress(posting)}")
                return True
            except Exception as e:
                print(f"Integrity check failed for {index_path}: {e}")
                return False

        def recursive_merge(indices: list, level: int = 1) -> str:
            """Recursively merge indices until one final index remains"""
            if len(indices) == 1:
                return indices[0]  # Base case: only one index left

            # Pair-wise merge
            next_level_outputs = []
            for i in range(0, len(indices), 2):
                pair = indices[i:i + 2]
                if len(pair) < 2:
                    next_level_outputs.append(pair[0])
                    continue

                output_path = f"Level{level}_Compressed_Merge_{i // 2 + 1}.vb"
                log_merge_operation(pair, f"Level {level}")

                # Merge the pair using the compressed index merger
                merged_index = self.index_builder.merge_compressed_indices_in_memory(
                    index1_path=pair[0],
                    index2_path=pair[1]
                )
                merged_index.write_compressed_index_to_file(output_path)
                next_level_outputs.append(output_path)

                # Validate the integrity of the merged index
                #if not validate_index_integrity(output_path):
                    #print(f"Error detected in merged index: {output_path}")
                    #raise ValueError(f"Failed integrity check for {output_path}")

                # Clean up
                del merged_index
                gc.collect()

            # Recurse with the next level outputs
            return recursive_merge(next_level_outputs, level + 1)

        # Initial partial compressed indices (assumed to exist)
        initial_indices = [f'Compressed_Index_{i}.vb' for i in range(1, 9)]
        print(f"Starting with {len(initial_indices)} initial compressed indices:")
        #for idx in initial_indices:
            #print(f"- {idx}: {get_file_size_mb(idx):.1f}MB")

            # Validate the integrity of each initial index
            #if not validate_index_integrity(idx):
                #print(f"Error detected in initial index: {idx}")
                #raise ValueError(f"Failed integrity check for {idx}")

        # Begin recursive merging
        final_path = recursive_merge(initial_indices)

        # Print final statistics
        print(f"\nMerge complete! Final index size: {get_file_size_mb(final_path):.1f}MB")
        if validate_index_integrity(final_path):
            print("Final index passed integrity check.")
        else:
            print("Final index failed integrity check.")

        final_index = self.index_builder.inverted_index.load_compressed_index_from_file(final_path)
        print(f"Terms in final index: {len(final_index.get_terms())}")

        # Optional: Clean up intermediate files
        # intermediate_files = [f for f in os.listdir() if f.startswith("Level") and f.endswith("Compressed_Merge")]
        # for file_path in intermediate_files:
            # if os.path.exists(file_path):
                # os.remove(file_path)
                # print(f"Cleaned up intermediate file: {file_path}")

    def test_build_partial_index(self):
        """Test partial index building with sampled documents"""
        start_time = time.time()
        expected_sample_size = 10000  # The sample size we expect from build_partial_index

        try:
            # Build the partial index
            self.index_builder.build_partial_index()
            build_time = time.time() - start_time

            # Get the final index
            index = self.index_builder.get_index()
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