import os
import struct
from typing import List
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

from InvertedIndex.InvertedIndex import InvertedIndex
from Utils.CollectionLoader import CollectionLoader
from Utils.Preprocessing import Preprocessing

class InvertedIndexBuilder:
    def __init__(
            self,
            collection_loader: CollectionLoader,
            preprocessing: Preprocessing,
            chunk_size: int = 1105228 # Number to have 8 chunks, which is optimal with my CPU.
    ) -> None:
        """
        Initialize the InvertedIndexBuilder.

        Args:
            collection_loader: Loader for the document collection
            preprocessing: Text preprocessing utilities
            chunk_size: Number of documents to process in each chunk
        """
        self.collection_loader = collection_loader
        self.preprocessing = preprocessing
        self.inverted_index = InvertedIndex()
        self.chunk_size = chunk_size

    def process_chunk(self, chunk: pd.DataFrame) -> InvertedIndex:
        """
        Process a single chunk of documents and return partial index.
        Ensures no duplicate postings for the same term in a document.

        Args:
            chunk: DataFrame containing document texts and IDs

        Returns:
            InvertedIndex containing the partial index for this chunk
        """
        if chunk is None or len(chunk) == 0:
            return InvertedIndex()

        try:
            chunk_index = InvertedIndex()
            # Preprocess the chunk
            tokens = self.preprocessing.vectorized_preprocess(chunk['text'])
            # Build local index for this chunk
            for doc_id, doc_tokens in zip(chunk['index'], tokens):
                # Use set to eliminate duplicates within the same document
                unique_tokens = set(doc_tokens)
                for token in unique_tokens:
                    if not token:  # Skip empty tokens
                        continue
                    chunk_index.add_posting(token, doc_id)
            del tokens
            gc.collect()
            return chunk_index

        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            return InvertedIndex()

    def build_full_index(self) -> None:
        """Build complete inverted index using chunk processing with compression."""
        try:
            total_docs = self.collection_loader.get_total_docs()
            total_chunks = (total_docs + self.chunk_size - 1) // self.chunk_size

            print(f"Processing {total_docs} documents in {total_chunks} chunks...")

            # Create list to store paths of partial compressed indices
            partial_indices_paths: List[str] = []
            chunks = self.collection_loader.process_chunks(chunk_size=self.chunk_size)
            counter = 0

            with tqdm(total=total_chunks) as pbar:
                for chunk in chunks:
                    # Process chunk into an inverted index
                    chunk_index = self.process_chunk(chunk)
                    counter += 1

                    # Construct a filename for the compressed index
                    chunk_index_path = f'Compressed_Index_{counter}.vb'
                    chunk_index.write_compressed_index_to_file(chunk_index_path)

                    # Store the path of the compressed index
                    partial_indices_paths.append(chunk_index_path)

                    # Free up memory after processing the chunk
                    del chunk
                    del chunk_index
                    gc.collect()
                    pbar.update(1)

            print("Merging indices...")
            # (Optional) Merge the partial compressed indices into a full index
            # self.inverted_index = self._merge_indices(partial_indices_paths)

            # Print some statistics about the final index
            total_terms = len(self.inverted_index.get_terms())
            print(f"Index built successfully with {total_terms} unique terms")

            # Delete the intermediate compressed index files
            #self._delete_partial_indices(partial_indices_paths)

        except Exception as e:
            print(f"Error building index: {str(e)}")
            raise


    @staticmethod
    def _delete_partial_indices(partial_indices_paths: List[str]) -> None:
        """Delete the intermediate index files on disk"""
        for index_path in partial_indices_paths:
            os.remove(index_path)
            print(f"Deleted intermediate index file: {index_path}")


    def build_partial_index(self) -> None:
        """
        Build a partial inverted index using a random sample of documents.
        This is useful for testing or quick analysis of the collection.
        """
        try:
            # Get sample of documents
            sample_df = self.collection_loader.sample_lines(num_lines=10000)

            print(f"Processing {len(sample_df)} sampled documents...")

            # Create new inverted index instance
            self.inverted_index = InvertedIndex()

            # Preprocess all texts at once
            tokens = self.preprocessing.vectorized_preprocess(sample_df['text'])

            # Add documents to index, ensuring no duplicates
            for doc_id, doc_tokens in zip(sample_df['index'], tokens):
                # Convert doc_id to int if necessary
                if not isinstance(doc_id, (int, np.int32, np.int64)):
                    doc_id = int(doc_id)

                # Create a set of unique tokens for this document
                unique_tokens = set(doc_tokens)

                # Add each unique token to the index
                for token in unique_tokens:
                    if token:  # Skip empty tokens
                        self.inverted_index.add_posting(token, doc_id)

            print(f"Partial index built with {len(self.inverted_index.get_terms())} unique terms")

        except Exception as e:
            print(f"Error building partial index: {str(e)}")
            raise



    def merge_two_indices(self, index1_path: str, index2_path: str, memory_efficient: bool = False) -> InvertedIndex:
        """
        Merges two inverted indices using either a fast in-memory approach or a memory-efficient streaming approach.

        Args:
            index1_path: Path to first index file
            index2_path: Path to second index file
            memory_efficient: If True, uses streaming approach for large files. If False, loads both indices in memory.

        Returns:
            InvertedIndex: Merged index containing unique postings from both indices
        """
        # Get file sizes
        size1 = os.path.getsize(index1_path)
        size2 = os.path.getsize(index2_path)

        # If either file is larger than 500MB or combined size > 800MB, use memory efficient method
        memory_threshold = 500 * 1024 * 1024  # 500MB
        combined_threshold = 900 * 1024 * 1024  # 900MB

        should_use_memory_efficient = (
                memory_efficient or
                size1 > memory_threshold or
                size2 > memory_threshold or
                (size1 + size2) > combined_threshold
        )

        if should_use_memory_efficient:
            print(
                f"Using memory-efficient merge for files: {size1 / 1024 / 1024:.1f}MB and {size2 / 1024 / 1024:.1f}MB")
            return self._merge_two_indices_streaming(index1_path, index2_path)
        else:
            print(
                f"Using fast in-memory merge for files: {size1 / 1024 / 1024:.1f}MB and {size2 / 1024 / 1024:.1f}MB")
            return self._merge_two_indices_in_memory(index1_path, index2_path)

    @staticmethod
    def _merge_two_indices_in_memory(index1_path: str, index2_path: str) -> InvertedIndex:
        """Original fast in-memory implementation"""
        final_index = InvertedIndex()

        # Load first index
        print("Loading first index into memory")
        index1 = InvertedIndex.load_from_file(index1_path)
        terms1 = set(index1.get_terms())

        # Load second index
        print("Loading second index into memory")
        index2 = InvertedIndex.load_from_file(index2_path)
        terms2 = set(index2.get_terms())

        # Process all unique terms
        all_terms = terms1.union(terms2)
        print(f"Merging {len(all_terms)} unique terms")

        for term in all_terms:
            postings1 = index1.get_postings(term) if term in terms1 else []
            postings2 = index2.get_postings(term) if term in terms2 else []

            doc_ids1 = {p.doc_id: p for p in postings1}
            unique_postings2 = [p for p in postings2 if p.doc_id not in doc_ids1]

            final_postings = list(doc_ids1.values()) + unique_postings2
            if final_postings:
                for posting in final_postings:
                    final_index.add_posting(term, posting.doc_id)

        del index1
        del index2
        gc.collect()

        return final_index

    @staticmethod
    def _merge_two_indices_streaming(index1_path: str, index2_path: str) -> InvertedIndex:
        """Memory-efficient streaming merge of two indexes saved in text format."""
        final_index = InvertedIndex()

        # First pass: Collect all unique terms
        print("First pass: Collecting terms")
        terms1 = InvertedIndex.read_index_terms(index1_path)
        terms2 = InvertedIndex.read_index_terms(index2_path)
        all_terms = sorted(terms1.union(terms2))
        print(f"Found {len(all_terms)} unique terms")

        # Second pass: Process terms in batches
        batch_size = 80000
        for i in range(0, len(all_terms), batch_size):
            batch_terms = set(all_terms[i:i + batch_size])
            print(f"Processing batch {i // batch_size + 1} of {(len(all_terms) + batch_size - 1) // batch_size}")

            # Process postings from the first index
            index1_batch = InvertedIndex.read_index_postings(index1_path, batch_terms)

            # Process postings from the second index and merge
            index2_batch = InvertedIndex.read_index_postings(index2_path, batch_terms)
            for term in batch_terms:
                postings1 = index1_batch.get(term, [])
                postings2 = index2_batch.get(term, [])

                doc_ids1 = {p.doc_id: p for p in postings1}
                unique_postings2 = [p for p in postings2 if p.doc_id not in doc_ids1]

                final_postings = list(doc_ids1.values()) + unique_postings2
                if final_postings:
                    final_index.add_postings(term, iter(final_postings))

            del index1_batch
            del index2_batch
            gc.collect()
        return final_index

    @staticmethod
    def _merge_compressed_postings(postings1: List[bytes], postings2: List[bytes]) -> List[bytes]:
        """Merges two lists of compressed postings and returns the merged compressed postings."""
        if not postings1:
            return postings2
        if not postings2:
            return postings1

        # Ensure we're working with actual compressed data
        valid_postings1 = [p for p in postings1 if isinstance(p, bytes) and len(p) > 0]
        valid_postings2 = [p for p in postings2 if isinstance(p, bytes) and len(p) > 0]

        # Decompress both posting lists
        doc_ids1 = []
        doc_ids2 = []

        for posting in valid_postings1:
            try:
                doc_ids1.extend(InvertedIndex.pfor_delta_decompress(posting))
            except Exception as e:
                print(f"Error decompressing posting1: {e}")
                continue

        for posting in valid_postings2:
            try:
                doc_ids2.extend(InvertedIndex.pfor_delta_decompress(posting))
            except Exception as e:
                print(f"Error decompressing posting2: {e}")
                continue

        # Merge and sort document IDs
        merged_doc_ids = sorted(set(doc_ids1).union(doc_ids2))

        if not merged_doc_ids:
            return []

        # Compress the merged document IDs
        try:
            return [InvertedIndex.pfor_delta_compress(merged_doc_ids)]
        except Exception as e:
            print(f"Error compressing merged doc_ids: {e}")
            return []

    def merge_compressed_indices_in_memory(self, index1_path: str, index2_path: str) -> InvertedIndex:
        """Merges two compressed indices while maintaining the defaultdict structure."""
        # Load the compressed indexes into memory
        print(f"Loading compressed index from {index1_path}")
        index1 = InvertedIndex.load_compressed_index_to_memory(filename=index1_path)
        print(f"Loading compressed index from {index2_path}")
        index2 = InvertedIndex.load_compressed_index_to_memory(filename=index2_path)

        merged_index = InvertedIndex()

        # Process terms from both indices
        all_terms = set(index1.get_terms()).union(index2.get_terms())
        total_terms = len(all_terms)

        print(f"Merging {total_terms} unique terms...")
        for i, term in enumerate(sorted(all_terms), 1):
            if i % 1000 == 0:
                print(f"Progress: {i}/{total_terms} terms processed")

            try:
                # Get compressed postings
                postings1 = index1.get_compressed_postings(term)
                postings2 = index2.get_compressed_postings(term)

                # Merge the postings
                merged_postings = self._merge_compressed_postings(postings1, postings2)

                if merged_postings:  # Only add if we have valid merged postings
                    merged_index.add_compressed_postings(term, merged_postings)

            except Exception as e:
                print(f"Error processing term '{term}': {e}")
                continue

        # Cleanup
        del index1
        del index2
        gc.collect()

        return merged_index

    def merge_and_write_compressed_indices(self, index1_path: str, index2_path: str, output_path: str) -> bool:
        """Merges two compressed indices and writes the result to a file."""
        try:
            merged_index = self.merge_compressed_indices_in_memory(index1_path, index2_path)

            # Verify the merged index has content before writing
            if not merged_index.get_terms():
                print("Warning: Merged index is empty!")
                return False

            print(f"Writing merged index to {output_path}")
            merged_index.write_compressed_index_to_file(output_path, is_compressed=True)

            # Verify the written file
            print("Verifying merged index...")
            try:
                verification_index = InvertedIndex.load_compressed_index_to_memory(output_path)
                if len(verification_index.get_terms()) > 0:
                    print("Verification successful!")
                    return True
                else:
                    print("Verification failed: Index is empty after loading")
                    return False
            except Exception as e:
                print(f"Verification failed: {e}")
                return False

        except Exception as e:
            print(f"Error during merge and write: {e}")
            return False

    @staticmethod
    def verify_index_integrity(filename: str) -> bool:
        """Verifies that an index file can be properly loaded and contains valid data."""
        try:
            with open(filename, 'rb') as f:
                while True:
                    # Read term length
                    term_length_bytes = f.read(2)
                    if not term_length_bytes:
                        break

                    # Verify term length
                    term_length = struct.unpack("H", term_length_bytes)[0]
                    if term_length <= 0 or term_length > 1000:  # reasonable max term length
                        raise ValueError(f"Invalid term length: {term_length}")

                    # Read and verify term
                    term_bytes = f.read(term_length)
                    try:
                        term = term_bytes.decode('utf-8')
                    except UnicodeDecodeError as e:
                        print(f"Invalid UTF-8 sequence in term at position {f.tell() - term_length}")
                        raise e

                    # Read compressed postings length
                    compressed_length_bytes = f.read(4)
                    if not compressed_length_bytes:
                        raise EOFError("Unexpected end of file while reading compressed length")

                    compressed_length = struct.unpack("I", compressed_length_bytes)[0]
                    if compressed_length <= 0 or compressed_length > 10_000_000:  # reasonable max compressed size
                        raise ValueError(f"Invalid compressed length: {compressed_length}")

                    # Read compressed data
                    compressed_data = f.read(compressed_length)
                    if len(compressed_data) != compressed_length:
                        raise EOFError("Incomplete compressed data")

                    # Try to decompress the data to verify it's valid
                    try:
                        InvertedIndex.pfor_delta_decompress(compressed_data)
                    except Exception as e:
                        print(f"Failed to decompress postings for term '{term}'")
                        raise e

            return True

        except Exception as e:
            print(f"Index integrity check failed: {e}")
            return False

    def get_index(self) -> InvertedIndex:
        return self.inverted_index
