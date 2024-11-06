import os
from typing import List
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from InvertedIndex.InvertedIndex import InvertedIndex
from InvertedIndex.Posting import Posting
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
        """Build complete inverted index using chunk processing"""
        try:
            total_docs = self.collection_loader.get_total_docs()
            total_chunks = (total_docs + self.chunk_size - 1) // self.chunk_size

            print(f"Processing {total_docs} documents in {total_chunks} chunks...")

            # Create list to store chunk indices paths
            partial_indices_paths: List[str] = []
            chunks = self.collection_loader.process_chunks(chunk_size=self.chunk_size)
            # Process chunks directly into separate inverted indices
            counter = 0
            with tqdm(total=total_chunks) as pbar:
                for chunk in chunks:
                    # Process chunk using existing method
                    chunk_index = self.process_chunk(chunk)
                    counter += 1
                    chunk_index_path = f'Index_{counter}'
                    chunk_index.write_to_file(chunk_index_path)
                    partial_indices_paths.append(chunk_index_path)
                    # Free up memory after processing the chunk
                    del chunk
                    del chunk_index
                    gc.collect()
                    pbar.update(1)

            print("Merging indices...")
            # Create new inverted index for the final result
            #self.inverted_index = self._merge_indices(partial_indices_paths)

            # Print some statistics about the final index
            total_terms = len(self.inverted_index.get_terms())
            print(f"Index built successfully with {total_terms} unique terms")

            # Delete the intermediate index files
            self._delete_partial_indices(partial_indices_paths)

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



    def _merge_two_indices(self, index1_path: str, index2_path: str, memory_efficient: bool = False) -> InvertedIndex:
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
    def _merge_two_indices_streaming(index1_path: str, index2_path: str) -> 'InvertedIndex':
        """
        Memory-efficient streaming implementation that merges two inverted indices.
        Uses text file format instead of pickle for better reliability and memory usage.
        """
        final_index = InvertedIndex()

        # Helper function to read terms and postings from a file
        def read_index_file(filename: str):
            terms_dict = {}
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    term = parts[0]
                    postings = []
                    for posting_data in parts[1:]:
                        doc_id, *payload = posting_data.split(":")
                        payload = ":".join(payload) if payload else None
                        postings.append(Posting(int(doc_id), payload))
                    terms_dict[term] = postings
            return terms_dict

        # First pass: Collect terms from both indices
        print("Reading and merging indices...")

        # Read both indices
        index1_terms = read_index_file(index1_path)
        index2_terms = read_index_file(index2_path)

        # Get all unique terms
        all_terms = sorted(set(index1_terms.keys()).union(set(index2_terms.keys())))
        print(f"Found {len(all_terms)} unique terms")

        # Merge postings for each term
        for term in all_terms:
            postings1 = index1_terms.get(term, [])
            postings2 = index2_terms.get(term, [])

            # Create a dictionary of doc_ids for efficient lookup
            doc_ids1 = {p.doc_id: p for p in postings1}

            # Add unique postings from index2
            unique_postings2 = [p for p in postings2 if p.doc_id not in doc_ids1]

            # Combine postings and add to final index
            final_postings = list(doc_ids1.values()) + unique_postings2
            for posting in final_postings:
                final_index.add_posting(term, posting.doc_id, posting.payload)

        return final_index

    def get_index(self) -> InvertedIndex:
        return self.inverted_index
