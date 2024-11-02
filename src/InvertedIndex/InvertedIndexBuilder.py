from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from InvertedIndex.InvertedIndex import InvertedIndex
from Utils.CollectionLoader import CollectionLoader
from Utils.Preprocessing import Preprocessing

class InvertedIndexBuilder:
    def __init__(
            self,
            collection_loader: CollectionLoader,
            preprocessing: Preprocessing,
            chunk_size: int = 500000
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

        Args:
            chunk: DataFrame containing document texts and IDs

        Returns:
            InvertedIndex containing the partial index for this chunk
        """
        if chunk is None or len(chunk) == 0:
            return InvertedIndex()

        # Check for required columns
        if 'text' not in chunk.columns or 'index' not in chunk.columns:
            raise ValueError("Chunk must contain 'text' and 'index' columns")

        try:
            chunk_index = InvertedIndex()

            # Preprocess the chunk
            tokens = self.preprocessing.vectorized_preprocess(chunk['text'])

            # Build local index for this chunk
            for doc_id, doc_tokens in zip(chunk['index'], tokens):
                if not isinstance(doc_id, (int, np.int32, np.int64)):
                    doc_id = int(doc_id)  # Try to convert to int if possible

                for token in doc_tokens:
                    if not token:  # Skip empty tokens
                        continue
                    chunk_index.add_posting(token, doc_id)

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
            chunks = self.collection_loader.process_chunks(chunk_size=self.chunk_size)

            # Process chunks directly into separate inverted indices
            with tqdm(total=total_chunks) as pbar:
                for chunk in chunks:
                    # Process chunk using existing method
                    chunk_index = self.process_chunk(chunk)
                    partial_indices.append(chunk_index)
                    pbar.update(1)

            print("Merging indices...")
            # Create new inverted index for the final result
            self.inverted_index = self._merge_indices(partial_indices)
        except Exception as e:
            print(f"Error building index: {str(e)}")
            raise

    @staticmethod
    def _merge_indices(partial_indices: List[InvertedIndex]) -> InvertedIndex:
        """Merge multiple inverted indices efficiently"""
        final_index = InvertedIndex()

        # Get all unique terms across all indices
        all_terms = set()
        for index in partial_indices:
            all_terms.update(index.get_terms())

        # Merge posting lists for each term
        with tqdm(total=len(all_terms)) as pbar:
            for term in all_terms:
                # Get all postings for this term across indices
                for partial_index in partial_indices:
                    postings = partial_index.get_postings(term)
                    if postings:
                        # Add each posting directly to final index
                        for posting in postings:
                            final_index.add_posting(term, posting.doc_id, posting.payload)
                pbar.update(1)

        return final_index

    def get_index(self) -> InvertedIndex:
        """Return the built inverted index"""
        return self.inverted_index