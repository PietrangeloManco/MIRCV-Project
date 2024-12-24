import gc
import os
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from DocumentTable.DocumentTable import DocumentTable
from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from InvertedIndex.InvertedIndex import InvertedIndex
from InvertedIndex.Merger import Merger
from Lexicon.Lexicon import Lexicon
from Utils.CollectionLoader import CollectionLoader
from Utils.MemoryProfile import MemoryProfile
from Utils.MemoryTrackingTools import MemoryTrackingTools
from Utils.Preprocessing import Preprocessing
from Utils.config import RESOURCES_PATH


class InvertedIndexBuilder:
    def __init__(
            self,
            collection_loader: CollectionLoader,
            preprocessing: Preprocessing,
            merger: Merger,
            lexicon: Lexicon,
            document_table: DocumentTable,
    ) -> None:
        """
        Initialize the InvertedIndexBuilder with necessary components.

        Args:
            collection_loader: Loader for the document collection.
            preprocessing: Text preprocessing utilities.
            merger: Component for merging partial indices.
            lexicon: Lexicon structure to be built alongside the inverted index.
            document_table: DocumentTable structure to be built alongside inverted index.
        """
        self.collection_loader = collection_loader
        self.preprocessing = preprocessing
        self.compressed_inverted_index = CompressedInvertedIndex()
        self.merger = merger
        self.lexicon = lexicon
        self.document_table = document_table
        # Memory tracking tools for dynamic chunking
        self.memory_tools = MemoryTrackingTools()
        # Global path to resources
        self.resources_path = RESOURCES_PATH

    def process_chunk(self, chunk: pd.DataFrame) -> InvertedIndex:
        """
        Process a chunk of documents into a partial index.

        Args:
            chunk(pd.DataFrame): DataFrame containing documents to process

        Returns:
            InvertedIndex: Partial inverted index for the chunk
        """
        if chunk is None or chunk.empty:
            return InvertedIndex()

        chunk_index = InvertedIndex()
        # Vectorized preprocessing for speed
        tokens_list = self.preprocessing.vectorized_preprocess(chunk['text'])

        doc_lengths = chunk['text'].str.split().str.len()

        # Update document table
        for doc_id, length in zip(chunk['index'], doc_lengths):
            self.document_table.add_document(doc_id, length)

        # Track document frequency for tokens for Lexicon
        doc_frequency_map = {}

        # Process tokens and update the index
        for doc_id, tokens in zip(chunk['index'], tokens_list):
            if not tokens:  # Skip empty documents
                continue

            # Count frequencies once per document
            token_freq_map = {}
            for token in tokens:
                if token:  # Skip empty tokens
                    token_freq_map[token] = token_freq_map.get(token, 0) + 1

                    # Track unique document IDs for each token
                    if token not in doc_frequency_map:
                        doc_frequency_map[token] = set()
                    doc_frequency_map[token].add(doc_id)

            # Update the inverted index
            for token, freq in token_freq_map.items():
                chunk_index.add_posting(token, doc_id, freq)

        # Update the lexicon with document frequency
        for token, doc_ids in doc_frequency_map.items():
            self.lexicon.add_term(token, document_frequency=len(doc_ids))

        return chunk_index

    def profile_memory_usage(self, sample_size: int) -> 'MemoryProfile':
        """
        Profile memory usage by processing a small sample to estimate
        total memory requirements including processing overhead.

        Args:
            sample_size(int): Number of documents to use for profiling.

        Returns:
            MemoryProfile: Memory usage estimates
        """
        # Measure initial memory state
        initial_memory = self.memory_tools.get_available_memory()
        total_memory = self.memory_tools.get_total_memory()
        gc.collect()  # Force clean up before profiling

        # Process sample and measure memory impact
        try:
            sample_chunk = self.collection_loader.process_single_chunk(0, sample_size)

            # Process the sample and measure total memory impact
            _ = self.process_chunk(sample_chunk)
            post_process_memory = self.memory_tools.get_available_memory()

            # Calculate memory usage per document including overhead
            total_memory_used = initial_memory - post_process_memory
            memory_per_doc = total_memory_used / sample_size

            # Calculate safe chunk size (targeting 80% of available memory)
            target_memory = total_memory * 0.8
            estimated_chunk_size = int(target_memory / memory_per_doc)

            return MemoryProfile(memory_per_doc=memory_per_doc, estimated_chunk_size=estimated_chunk_size)

        finally:
            # Clean up profiling data
            gc.collect()

    def _process_and_save_chunk(self, chunk: DataFrame, index_num: int) -> str:
        """
        Process a chunk and save its partial compressed index.

        Args:
            chunk(DataFrame): Chunk of documents to process.
            index_num(int): Ordinal number to track the partial indices building.

        Returns:
            str: The path to the partial inverted index.
        """
        chunk_index = self.process_chunk(chunk)
        index_path = self.resources_path + f"Compressed_Index_{index_num}.vb"
        chunk_index.write_index_compressed_to_file(index_path)

        # Free up the memory used and force garbage collection
        del chunk_index
        gc.collect()

        return index_path

    def build_partial_indices(self, use_static_chunk_size: bool = False, static_chunk_size: Optional[int] = None) -> \
            List[str]:
        """
        Build partial compressed indices with dynamic chunk sizing based on memory profiling or fixed chunk size.

        Args:
            use_static_chunk_size: Whether to use a static chunk size. Default is False.
            static_chunk_size: The static chunk size to use if `use_static_chunk_size` is True.

        Returns:
            List[str]: List of paths to partial indices.
        """
        if use_static_chunk_size:
            if not static_chunk_size:
                raise ValueError("Static chunk size must be provided when using static chunk size.")
            return self._process_with_static_chunk_size(static_chunk_size)

        total_docs = self.collection_loader.get_total_docs()
        print(f"Processing {total_docs} documents...")

        # Run memory profiling on a small sample to estimate total memory impact
        sample_size = min(10000, total_docs)  # Use 10000 docs or fewer for estimation
        initial_memory = self.memory_tools.get_available_memory()
        print(f"Initial available memory: {initial_memory} bytes")

        # Profile memory usage including processing overhead
        memory_profile = self.profile_memory_usage(sample_size)
        if memory_profile.estimated_chunk_size <= 0:
            raise RuntimeError("Not enough memory to process even a minimal chunk")

        print(f"Memory profiling results:")
        print(f"- Memory per document (with overhead): {memory_profile.memory_per_doc / 1024 / 1024:.2f} MB")
        print(f"- Recommended chunk size: {memory_profile.estimated_chunk_size} documents")
        # Safe limit for the initial run, in which the profiler tends to underestimate memory impact.
        if memory_profile.estimated_chunk_size > 1000000:
            print("Using default chunk size of 1.0 million documents")
        # Process the collection in chunks
        partial_indices_paths: List[str] = []
        chunk_start = 0

        while chunk_start < total_docs:
            # Adjust chunk size based on available memory
            current_available = self.memory_tools.get_available_memory()
            current_chunk_size = min(
                min(memory_profile.estimated_chunk_size, 1000000),
                total_docs - chunk_start
            )

            if current_available < memory_profile.memory_per_doc * current_chunk_size:
                # Reduce chunk size if memory is tight
                current_chunk_size = int(current_available * 0.8 / memory_profile.memory_per_doc)
                print(f"Adjusting chunk size to {current_chunk_size} due to memory constraints")

            current_chunk = self.collection_loader.process_single_chunk(
                chunk_start,
                current_chunk_size
            )

            # Process and save the chunk
            index_path = self.resources_path + f"Compressed_Index_{len(partial_indices_paths) + 1}.vb"
            chunk_index = self.process_chunk(current_chunk)
            chunk_index.write_index_compressed_to_file(index_path)
            partial_indices_paths.append(index_path)

            # Clean up memory
            del current_chunk
            del chunk_index
            gc.collect()

            chunk_start += current_chunk_size

        return partial_indices_paths

    def build_full_index(self, use_static_chunk_size: bool = False, static_chunk_size: Optional[int] = None) -> None:
        """
        Build and save the complete compressed inverted index, lexicon
        and document table with an optional static chunk size.

        Args:
            use_static_chunk_size(bool): Whether to use or not the static chunking option. Default is False.
            static_chunk_size(Optional[int]): If the static chunking is used, size of the chunk.
        """
        try:
            partial_indices_paths = self.build_partial_indices(use_static_chunk_size, static_chunk_size)

            # Save auxiliary structures
            self.lexicon.write_to_file(self.resources_path + "Lexicon")
            self.document_table.write_to_file(self.resources_path + "DocumentTable")

            # Merge indices
            print("Merging indices...")
            self.compressed_inverted_index = self.merger.merge_multiple_compressed_indices(
                partial_indices_paths
            )

            # Save final index and clean up
            self.compressed_inverted_index.write_compressed_index_to_file(self.resources_path + "InvertedIndex")
            self._delete_partial_indices(partial_indices_paths)

            print(f"Index built successfully with {len(self.compressed_inverted_index.get_terms())} unique terms.")

        except Exception as e:
            print(f"Error building full index: {str(e)}")
            raise

    def build_partial_index(self, sample_size: int = 10000) -> None:
        """
        Build a partial compressed inverted index for testing.

        Args:
            sample_size(int): Number of documents to sample. Default 10000.
        """
        try:
            sample_df = self.collection_loader.sample_lines(sample_size)
            index_path = self._process_and_save_chunk(sample_df, 1)

            self.compressed_inverted_index = (
                self.compressed_inverted_index.load_compressed_index_to_memory(index_path)
            )

            # Save auxiliary structures
            self.lexicon.write_to_file(self.resources_path + "partial_lexicon.txt")
            self.document_table.write_to_file(self.resources_path + "partial_document_table.txt")

            print(f"Partial index built with {len(self.compressed_inverted_index.get_terms())} unique terms.")

        except Exception as e:
            print(f"Error building partial index: {str(e)}")
            raise

    def _process_with_static_chunk_size(self, static_chunk_size: int) -> List[str]:
        """
        Process and build indices using a static chunk size.

        Args:
            static_chunk_size(int): The static chunk size to use for processing.

        Returns:
            List[str]: List of paths to the partial indices.
        """
        total_docs = self.collection_loader.get_total_docs()
        print(f"Processing {total_docs} documents with static chunk size of {static_chunk_size}...")

        partial_indices_paths: List[str] = []

        # Iterate through chunks
        for chunk in self.collection_loader.process_chunks(static_chunk_size):
            # Process and save the chunk
            index_path = self.resources_path + f"Compressed_Index_{len(partial_indices_paths) + 1}.vb"
            chunk_index = self.process_chunk(chunk)
            chunk_index.write_index_compressed_to_file(index_path)
            partial_indices_paths.append(index_path)

            # Clean up memory after processing the chunk
            del chunk
            del chunk_index
            gc.collect()

        return partial_indices_paths

    @staticmethod
    def _delete_partial_indices(partial_indices_paths: List[str]) -> None:
        """
        Clean up intermediate index files.

        Args:
            partial_indices_paths(List[str]): The list of paths to the partial indexes to delete.
        """
        for path in partial_indices_paths:
            os.remove(path)
            print(f"Deleted intermediate index file: {path}")

    def get_index(self) -> CompressedInvertedIndex:
        """
        Getter for the built compressed inverted index.
        """
        return self.compressed_inverted_index

    def get_lexicon(self) -> Lexicon:
        """
        Getter for the built lexicon.
        """
        return self.lexicon

    def get_document_table(self) -> DocumentTable:
        """
        Getter for the built document table.
        """
        return self.document_table
