import os
from typing import List
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from InvertedIndex.InvertedIndex import InvertedIndex
from Utils.CollectionLoader import CollectionLoader
from Utils.Preprocessing import Preprocessing


class InvertedIndexBuilder:
    def __init__(self, collection_loader: CollectionLoader, preprocessing: Preprocessing, chunk_size: int = 1105228) -> None:
        """
        Initialize the InvertedIndexBuilder.

        Args:
            collection_loader: Loader for the document collection.
            preprocessing: Text preprocessing utilities.
            chunk_size: Number of documents to process in each chunk.
        """
        self.collection_loader = collection_loader
        self.preprocessing = preprocessing
        self.compressed_inverted_index = CompressedInvertedIndex()
        self.chunk_size = chunk_size

    def process_chunk(self, chunk: pd.DataFrame) -> InvertedIndex:
        """
        Process a single chunk of documents and return a partial index.
        Ensures no duplicate postings for the same term in a document.

        Args:
            chunk: DataFrame containing document texts and IDs.

        Returns:
            InvertedIndex containing the partial index for this chunk.
        """
        if chunk is None or chunk.empty:
            return InvertedIndex()

        try:
            chunk_index = InvertedIndex()
            # Preprocess the chunk
            tokens_list = self.preprocessing.vectorized_preprocess(chunk['text'])

            for doc_id, tokens in zip(chunk['index'], tokens_list):
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    if token:  # Skip empty tokens
                        chunk_index.add_posting(token, doc_id)

            return chunk_index

        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            return InvertedIndex()

    def build_full_index(self) -> None:
        """Build a complete inverted index using chunk processing with compression."""
        try:
            total_docs = self.collection_loader.get_total_docs()
            total_chunks = (total_docs + self.chunk_size - 1) // self.chunk_size

            print(f"Processing {total_docs} documents in {total_chunks} chunks...")

            partial_indices_paths: List[str] = []
            chunks = self.collection_loader.process_chunks(self.chunk_size)
            counter = 0

            with tqdm(total=total_chunks) as pbar:
                for chunk in chunks:
                    chunk_index = self.process_chunk(chunk)
                    counter += 1

                    # Write the partial compressed index to a file
                    chunk_index_path = f'Compressed_Index_{counter}.vb'
                    chunk_index.write_index_compressed_to_file(chunk_index_path)
                    partial_indices_paths.append(chunk_index_path)

                    # Clean up memory
                    del chunk
                    del chunk_index
                    gc.collect()
                    pbar.update(1)

            print("Merging indices...")
            # Merge all partial indices if needed
            # self.inverted_index = self._merge_partial_indices(partial_indices_paths)

            total_terms = len(self.compressed_inverted_index.get_terms())
            print(f"Index built successfully with {total_terms} unique terms.")

            # Optionally, delete intermediate files
            # self._delete_partial_indices(partial_indices_paths)

        except Exception as e:
            print(f"Error building index: {str(e)}")
            raise

    @staticmethod
    def _delete_partial_indices(partial_indices_paths: List[str]) -> None:
        """Delete the intermediate index files from disk."""
        for path in partial_indices_paths:
            os.remove(path)
            print(f"Deleted intermediate index file: {path}")

    def build_partial_index(self) -> None:
        """Build a partial inverted index using a sample of documents for testing or analysis."""
        try:
            sample_df = self.collection_loader.sample_lines(10000)
            print(f"Processing {len(sample_df)} sampled documents...")

            self.compressed_inverted_index = InvertedIndex()
            tokens_list = self.preprocessing.vectorized_preprocess(sample_df['text'])

            for doc_id, tokens in zip(sample_df['index'], tokens_list):
                if not isinstance(doc_id, (int, np.int32, np.int64)):
                    doc_id = int(doc_id)
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    if token:
                        self.compressed_inverted_index.add_posting(token, doc_id)

            print(f"Partial index built with {len(self.compressed_inverted_index.get_terms())} unique terms.")

        except Exception as e:
            print(f"Error building partial index: {str(e)}")
            raise

    def get_index(self) -> CompressedInvertedIndex:
        """Get the built inverted index."""
        return self.compressed_inverted_index
