import os
from typing import List, Optional
import pandas as pd
import gc
from pandas import DataFrame
from tqdm import tqdm

from DocumentTable.DocumentTable import DocumentTable
from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from InvertedIndex.InvertedIndex import InvertedIndex
from InvertedIndex.Merger import Merger
from Lexicon.Lexicon import Lexicon
from Utils.CollectionLoader import CollectionLoader
from Utils.MemoryProfile import MemoryProfile
from Utils.MemoryTrackingTools import MemoryTrackingTools
from Utils.Preprocessing import Preprocessing


class InvertedIndexBuilder:
    def __init__(
            self,
            collection_loader: CollectionLoader,
            preprocessing: Preprocessing,
            merger: Merger,
            lexicon: Lexicon,
            document_table: DocumentTable,
            initial_chunk_size: int = 500000,
    ) -> None:
        """
        Initialize the InvertedIndexBuilder with necessary components.

        Args:
            collection_loader: Loader for the document collection
            preprocessing: Text preprocessing utilities
            merger: Component for merging indices
            lexicon: Lexicon component for term management
            document_table: Document metadata storage
            initial_chunk_size: Initial number of documents per chunk
        """
        self.collection_loader = collection_loader
        self.preprocessing = preprocessing
        self.compressed_inverted_index = CompressedInvertedIndex()
        self.initial_chunk_size = initial_chunk_size
        self.merger = merger
        self.lexicon = lexicon
        self.document_table = document_table
        self.memory_tools = MemoryTrackingTools()

    def process_chunk(self, chunk: pd.DataFrame) -> InvertedIndex:
        """
        Process a chunk of documents into a partial index.

        Args:
            chunk: DataFrame containing documents to process

        Returns:
            Partial inverted index for the chunk
        """
        if chunk is None or chunk.empty:
            return InvertedIndex()

        chunk_index = InvertedIndex()
        tokens_list = self.preprocessing.vectorized_preprocess(chunk['text'])

        # Update document table with lengths
        doc_lengths = chunk['text'].str.split().str.len()
        for doc_id, length, idx in zip(chunk['index'], doc_lengths, range(len(chunk))):
            self.document_table.add_document(doc_id, length, idx)

        # Build frequency maps and update lexicon
        for doc_id, tokens, idx in zip(chunk['index'], tokens_list, range(len(chunk))):
            if not tokens:
                continue

            token_freq_map = {}
            for token in tokens:
                if token:
                    token_freq_map[token] = token_freq_map.get(token, 0) + 1

            for token, freq in token_freq_map.items():
                chunk_index.add_posting(token, doc_id, freq)
                self.lexicon.add_term(token, position=idx, term_frequency=freq)

        return chunk_index

    def profile_memory_usage(self, sample_size: int) -> 'MemoryProfile':
        """
        Profile memory usage by processing a small sample to estimate
        total memory requirements including processing overhead.

        Args:
            sample_size: Number of documents to use for profiling

        Returns:
            MemoryProfile with memory usage estimates
        """
        # Measure initial memory state
        initial_memory = self.memory_tools.get_available_memory()
        total_memory = self.memory_tools.get_total_memory()
        gc.collect()  # Clean up before profiling

        # Process sample and measure memory impact
        try:
            sample_chunk = self.collection_loader.process_single_chunk(0, sample_size)

            # Process the sample and measure total memory impact
            _ = self.process_chunk(sample_chunk)
            post_process_memory = self.memory_tools.get_available_memory()

            # Calculate memory usage per document including overhead
            total_memory_used = initial_memory - post_process_memory
            memory_per_doc = total_memory_used / sample_size

            # Calculate safe chunk size (targeting 90% of available memory)
            target_memory = total_memory * 0.9
            estimated_chunk_size = int(target_memory / memory_per_doc)

            return MemoryProfile(memory_per_doc=memory_per_doc, estimated_chunk_size=estimated_chunk_size)

        finally:
            # Clean up profiling data
            gc.collect()

    def _process_and_save_chunk(self, chunk: DataFrame, index_num: int) -> str:
        """Process a chunk and save its compressed index."""
        chunk_index = self.process_chunk(chunk)
        index_path = f'Compressed_Index_{index_num}.vb'
        chunk_index.write_index_compressed_to_file(index_path)

        del chunk_index
        gc.collect()

        return index_path

    def build_partial_indices(self, sample_df: Optional[DataFrame] = None) -> List[str]:
        """
        Build partial compressed indices with dynamic chunk sizing based on memory profiling.
        Estimates total memory usage including processing overhead using a small sample.

        Args:
            sample_df: Optional DataFrame for testing/development

        Returns:
            List of paths to partial indices
        """
        if sample_df is not None:
            return [self._process_and_save_chunk(sample_df, 1)]

        total_docs = self.collection_loader.get_total_docs()
        print(f"Processing {total_docs} documents...")

        # Run memory profiling on a small sample to estimate total memory impact
        sample_size = min(10000, total_docs)  # Use 1000 docs or fewer for estimation
        initial_memory = self.memory_tools.get_available_memory()
        print(f"Initial available memory: {initial_memory} bytes")

        # Profile memory usage including processing overhead
        memory_profile = self.profile_memory_usage(sample_size)
        if memory_profile.estimated_chunk_size <= 0:
            raise RuntimeError("Not enough memory to process even a minimal chunk")

        print(f"Memory profiling results:")
        print(f"- Memory per document (with overhead): {memory_profile.memory_per_doc / 1024 / 1024:.2f} MB")
        print(f"- Recommended chunk size: {memory_profile.estimated_chunk_size} documents")

        # Process the collection in chunks
        partial_indices_paths: List[str] = []
        chunk_start = 0

        with tqdm(total=total_docs) as pbar:
            while chunk_start < total_docs:
                # Adjust chunk size based on available memory
                current_available = self.memory_tools.get_available_memory()
                current_chunk_size = min(
                    memory_profile.estimated_chunk_size,
                    total_docs - chunk_start
                )

                if current_available < memory_profile.memory_per_doc * current_chunk_size:
                    # Reduce chunk size if memory is tight
                    current_chunk_size = int(current_available * 0.9 / memory_profile.memory_per_doc)
                    print(f"Adjusting chunk size to {current_chunk_size} due to memory constraints")

                current_chunk = self.collection_loader.process_single_chunk(
                    chunk_start,
                    current_chunk_size
                )

                # Process and save the chunk
                index_path = f'Compressed_Index_{len(partial_indices_paths) + 1}.vb'
                chunk_index = self.process_chunk(current_chunk)
                chunk_index.write_index_compressed_to_file(index_path)
                partial_indices_paths.append(index_path)

                # Clean up memory
                del current_chunk
                del chunk_index
                gc.collect()

                chunk_start += current_chunk_size
                pbar.update(current_chunk_size)

        return partial_indices_paths

    def build_full_index(self) -> None:
        """Build and save the complete inverted index."""
        try:
            partial_indices_paths = self.build_partial_indices()

            # Save auxiliary structures
            self.lexicon.write_to_file("Lexicon")
            self.document_table.write_to_file("DocumentTable")

            # Merge indices
            print("Merging indices...")
            self.compressed_inverted_index = self.merger.merge_multiple_compressed_indices(
                partial_indices_paths
            )

            # Save final index and clean up
            self.compressed_inverted_index.write_compressed_index_to_file("InvertedIndex")
            self._delete_partial_indices(partial_indices_paths)

            print(f"Index built successfully with {len(self.compressed_inverted_index.get_terms())} unique terms.")

        except Exception as e:
            print(f"Error building full index: {str(e)}")
            raise

    def build_partial_index(self, sample_size: int = 10000) -> None:
        """
        Build a sample index for testing/development.

        Args:
            sample_size: Number of documents to sample
        """
        try:
            sample_df = self.collection_loader.sample_lines(sample_size)
            index_path = self.build_partial_indices(sample_df)[0]

            self.compressed_inverted_index = (
                self.compressed_inverted_index.load_compressed_index_to_memory(index_path)
            )

            # Save auxiliary structures
            self.lexicon.write_to_file("partial_lexicon.txt")
            self.document_table.write_to_file("partial_document_table.txt")

            print(f"Partial index built with {len(self.compressed_inverted_index.get_terms())} unique terms.")

        except Exception as e:
            print(f"Error building partial index: {str(e)}")
            raise

    @staticmethod
    def _delete_partial_indices(partial_indices_paths: List[str]) -> None:
        """Clean up intermediate index files."""
        for path in partial_indices_paths:
            os.remove(path)
            print(f"Deleted intermediate index file: {path}")

    def get_index(self) -> CompressedInvertedIndex:
        """Return the built inverted index."""
        return self.compressed_inverted_index

    def get_lexicon(self) -> Lexicon:
        """Return the lexicon component."""
        return self.lexicon

    def get_document_table(self) -> DocumentTable:
        """Return the document table component."""
        return self.document_table