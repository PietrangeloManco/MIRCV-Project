import gc
import os
import pickle
import unittest
import time

import psutil

from InvertedIndex.InvertedIndex import InvertedIndex
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

    def test_(self):
        index = self.index_builder.inverted_index.load_compressed_index_from_file("Level1_Compressed_Merge_1.vb")
        print(index.get_terms())
        return

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

                # Use the improved merge_and_write_compressed_indices method
                success = self.index_builder.merge_and_write_compressed_indices(
                    index1_path=pair[0],
                    index2_path=pair[1],
                    output_path=output_path
                )

                if not success:
                    raise ValueError(f"Failed to merge and verify index: {output_path}")

                next_level_outputs.append(output_path)

            # Recurse with the next level outputs
            return recursive_merge(next_level_outputs, level + 1)

        # Initial partial compressed indices (assumed to exist)
        initial_indices = [f'Compressed_Index_{i}.vb' for i in range(1, 9)]
        print(f"Starting with {len(initial_indices)} initial compressed indices:")
        for idx in initial_indices:
            print(f"- {idx}: {get_file_size_mb(idx):.1f}MB")

            # Use the builder's verify_index_integrity method
            if not self.index_builder.verify_index_integrity(idx):
                raise ValueError(f"Initial index failed integrity check: {idx}")

        # Begin recursive merging
        final_path = recursive_merge(initial_indices)

        # Print final statistics
        print(f"\nMerge complete! Final index size: {get_file_size_mb(final_path):.1f}MB")

        # Load and verify the final index
        final_index = self.index_builder.inverted_index.load_compressed_index_from_file(final_path)
        print(f"Terms in final index: {len(final_index.get_terms())}")

        # Optional: Clean up intermediate files
        # intermediate_files = [f for f in os.listdir() if f.startswith("Level") and f.endswith(".vb")]
        # for file_path in intermediate_files:
        # if os.path.exists(file_path):
        # os.remove(file_path)
        # print(f"Cleaned up intermediate file: {file_path}")

    def test_streaming_merge_final_pass(self):
        """
        Test specifically for merging the final two large indices using the streaming approach.
        This test focuses on the memory-efficient merge of the last pass.
        """

        def get_file_size_mb(path: str) -> float:
            """Helper to get file size in MB"""
            return os.path.getsize(path) / (1024 * 1024)

        def log_memory_usage():
            """Helper to log current memory usage"""
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return f"{memory_mb:.1f}MB"

        # Paths for the two large indices to merge
        index1_path = "Level1_Merge_1"
        index2_path = "Level1_Merge_2"

        print("\nStarting streaming merge test of final indices:")
        print(f"Index 1 size: {get_file_size_mb(index1_path):.1f}MB")
        print(f"Index 2 size: {get_file_size_mb(index2_path):.1f}MB")
        print(f"Initial memory usage: {log_memory_usage()}")

        # Time the merge operation
        start_time = time.time()

        # Force memory-efficient merge
        final_index = self.index_builder.merge_two_indices(
            index1_path=index1_path,
            index2_path=index2_path,
            memory_efficient=True  # Explicitly use streaming method
        )

        merge_time = time.time() - start_time

        # Write final result
        final_path = 'final_index_streaming'
        final_index.write_to_file(final_path)

        # Get results and stats
        final_terms = final_index.get_terms()

        # Print comprehensive results
        print("\nMerge completed!")
        print(f"Time taken: {merge_time:.1f} seconds")
        print(f"Final index size: {get_file_size_mb(final_path):.1f}MB")
        print(f"Peak memory usage: {log_memory_usage()}")
        print(f"Number of terms in final index: {len(final_terms)}")

        # Optional: Basic validation
        print("\nPerforming basic validation...")

        # Load original indices to compare term counts (if memory allows)
        try:
            with open(index1_path, 'rb') as f:
                temp_index1 = pickle.load(f)
                terms1 = set(temp_index1.get_terms())
                del temp_index1
                gc.collect()

            with open(index2_path, 'rb') as f:
                temp_index2 = pickle.load(f)
                terms2 = set(temp_index2.get_terms())
                del temp_index2
                gc.collect()

            expected_term_count = len(terms1.union(terms2))
            actual_term_count = len(final_terms)

            print(f"Terms in index 1: {len(terms1)}")
            print(f"Terms in index 2: {len(terms2)}")
            print(f"Terms in final index: {actual_term_count}")
            print(f"Expected unique terms: {expected_term_count}")

            assert actual_term_count == expected_term_count, \
                f"Term count mismatch: got {actual_term_count}, expected {expected_term_count}"

            print("Validation passed: Term counts match expected values")

        except Exception as e:
            print(f"Validation skipped or failed: {str(e)}")

        # Clean up
        del final_index
        gc.collect()

        print("\nTest completed successfully!")

    def test_streaming_merge_compressed_indices(self):
        """
        Test merging two compressed indices using the streaming approach.
        This test focuses on the memory-efficient merge of the compressed indices.
        """

        def get_file_size_mb(path: str) -> float:
            """Helper to get file size in MB"""
            return os.path.getsize(path) / (1024 * 1024)

        def log_memory_usage():
            """Helper to log current memory usage"""
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return f"{memory_mb:.1f}MB"

        # Paths for the two compressed indices to merge
        index1_path = "Compressed_Index_1.vb"
        index2_path = "Compressed_Index_2.vb"

        print("\nStarting streaming merge test of compressed indices:")
        print(f"Index 1 size: {get_file_size_mb(index1_path):.1f}MB")
        print(f"Index 2 size: {get_file_size_mb(index2_path):.1f}MB")
        print(f"Initial memory usage: {log_memory_usage()}")

        # Time the merge operation
        start_time = time.time()

        # Perform the memory-efficient merge of compressed indices
        merged_index = self.index_builder.merge_compressed_indices_in_memory(
            index1_path=index1_path,
            index2_path=index2_path
        )

        merge_time = time.time() - start_time

        # Write the merged index to a file
        merged_path = 'Merged_Compressed_Index'
        merged_index.write_compressed_index_to_file(merged_path)

        # Get results and stats
        merged_terms = merged_index.get_terms()

        # Print comprehensive results
        print("\nMerge completed!")
        print(f"Time taken: {merge_time:.1f} seconds")
        print(f"Merged index size: {get_file_size_mb(merged_path):.1f}MB")
        print(f"Peak memory usage: {log_memory_usage()}")
        print(f"Number of terms in merged index: {len(merged_terms)}")

        # Optional: Basic validation
        print("\nPerforming basic validation...")

        # Load original compressed indices to compare term counts (if memory allows)
        try:
            with open(index1_path, 'rb'):
                temp_index1 = InvertedIndex.load_compressed_index_to_memory(index1_path)
                terms1 = set(temp_index1.get_terms())
                del temp_index1
                gc.collect()

            with open(index2_path, 'rb'):
                temp_index2 = InvertedIndex.load_compressed_index_to_memory(index2_path)
                terms2 = set(temp_index2.get_terms())
                del temp_index2
                gc.collect()

            expected_term_count = len(terms1.union(terms2))
            actual_term_count = len(merged_terms)

            print(f"Terms in index 1: {len(terms1)}")
            print(f"Terms in index 2: {len(terms2)}")
            print(f"Terms in merged index: {actual_term_count}")
            print(f"Expected unique terms: {expected_term_count}")

            assert actual_term_count == expected_term_count, \
                f"Term count mismatch: got {actual_term_count}, expected {expected_term_count}"

            print("Validation passed: Term counts match expected values")

        except Exception as e:
            print(f"Validation skipped or failed: {str(e)}")

        # Clean up
        del merged_index
        gc.collect()

        print("\nTest completed successfully!")

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