from typing import List, Any
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
            chunk_size: int = 1105228
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
            print("Initialized chunk...")
            # Preprocess the chunk
            tokens = self.preprocessing.vectorized_preprocess(chunk['text'])
            print("Preprocessed tokens...")
            # Build local index for this chunk
            for doc_id, doc_tokens in zip(chunk['index'], tokens):
                # Use set to eliminate duplicates within the same document
                unique_tokens = set(doc_tokens)
                for token in unique_tokens:
                    if not token:  # Skip empty tokens
                        continue
                    chunk_index.add_posting(token, doc_id)
            self.free_memory(tokens)
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

            # Create list to store chunk indices
            partial_indices: List[InvertedIndex] = []
            print("Created partial indices...")
            chunks = self.collection_loader.process_chunks(chunk_size=self.chunk_size)
            print("Loaded collection...")
            # Process chunks directly into separate inverted indices
            with tqdm(total=total_chunks) as pbar:
                for chunk in chunks:
                    # Process chunk using existing method
                    print("Processing chunk...")
                    chunk_index = self.process_chunk(chunk)
                    print("got chunk")
                    partial_indices.append(chunk_index)
                    # Free up memory after processing the chunk
                    self.free_memory(chunk)
                    self.free_memory(chunk_index)
                    pbar.update(1)

            print("Merging indices...")
            # Create new inverted index for the final result
            self.inverted_index = self._merge_indices(partial_indices)

            # Print some statistics about the final index
            total_terms = len(self.inverted_index.get_terms())
            print(f"Index built successfully with {total_terms} unique terms")

        except Exception as e:
            print(f"Error building index: {str(e)}")
            raise

    def build_partial_index(self) -> None:
        """
        Build a partial inverted index using a random sample of documents.
        This is useful for testing or quick analysis of the collection.
        """
        try:
            # Get sample of documents
            print("Sampling documents...")
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

    @staticmethod
    def _merge_indices(partial_indices: List[InvertedIndex]) -> InvertedIndex:
        """
        Merge multiple inverted indices efficiently while avoiding duplicate postings.
        Uses a dictionary to track already added document IDs for each term.
        """
        final_index = InvertedIndex()

        # Get all unique terms across all indices
        all_terms = set()
        for index in partial_indices:
            all_terms.update(index.get_terms())

        # Track seen doc_ids for each term to avoid duplicates
        seen_docs = {}

        # Merge posting lists for each term
        with tqdm(total=len(all_terms)) as pbar:
            for term in all_terms:
                seen_docs[term] = set()  # Initialize set for this term

                # Get all postings for this term across indices
                for partial_index in partial_indices:
                    postings = partial_index.get_postings(term)
                    if postings:
                        # Only add postings we haven't seen before
                        for posting in postings:
                            if posting.doc_id not in seen_docs[term]:
                                final_index.add_posting(term, posting.doc_id, posting.payload)
                                seen_docs[term].add(posting.doc_id)
                pbar.update(1)

        return final_index

    @staticmethod
    def free_memory(_: Any) -> None:
        """
        Free up memory by clearing the chunk and preprocessed tokens.

        Args:
            _: memory to be free up
        """
        del _
        gc.collect()

    def get_index(self) -> InvertedIndex:
        return self.inverted_index
