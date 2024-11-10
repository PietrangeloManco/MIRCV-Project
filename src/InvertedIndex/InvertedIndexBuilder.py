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
from Utils.Preprocessing import Preprocessing


class InvertedIndexBuilder:
    def __init__(self, collection_loader: CollectionLoader, preprocessing: Preprocessing,
                 merger: Merger, lexicon: Lexicon, document_table: DocumentTable,
                 chunk_size: int = 1105228) -> None:
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
        self.merger = merger
        self.lexicon = lexicon
        self.document_table = document_table

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

            # Create a temporary 'tokens' column to store the tokenized text
            chunk['tokens'] = tokens_list

            for _, row in chunk.iterrows():
                doc_id = row['index']
                text = row['text']  # Assuming 'text' is preprocessed or tokenized
                tokens = row['tokens']  # Tokenized words
                token_freq_map = {}  # Map to store token frequency for the document

                for token in tokens:
                    if token:  # Skip empty tokens
                        token_freq_map[token] = token_freq_map.get(token, 0) + 1

                # Add postings for each token in the document
                for token, freq in token_freq_map.items():
                    chunk_index.add_posting(token, doc_id, freq)

                # Now update lexicon and document table (these should be updated after processing tokens)
                length = len(text.split())  # Length of the document in terms
                offset = chunk.index[chunk.index == _].tolist()[0]  # Offset of the document in the chunk

                # Add to Document Table
                self.document_table.add_document(doc_id, length, offset)

                # Update Lexicon
                for term in tokens:
                    self.lexicon.add_term(term, position=offset, term_frequency=tokens.count(term))

            return chunk_index

        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            return InvertedIndex()

    def build_partial_indices(self, sample_df: Optional[DataFrame] = None) -> List[str]:
        """Build partial compressed indices in chunks and return their file paths."""
        try:
            # Determine whether we're processing the entire collection or just a sample
            if sample_df is None:
                # Full collection: process chunks
                total_docs = self.collection_loader.get_total_docs()
                total_chunks = (total_docs + self.chunk_size - 1) // self.chunk_size
                print(f"Processing {total_docs} documents in {total_chunks} chunks...")
                chunks = self.collection_loader.process_chunks(self.chunk_size)
            else:
                # Partial index: single chunk (i.e., sample_df)
                total_docs = len(sample_df)
                total_chunks = 1
                print(f"Processing {total_docs} documents in {total_chunks} chunk...")
                chunks = [sample_df]  # Wrap sample_df in a list to treat as a single chunk

            partial_indices_paths: List[str] = []

            # Initialize lexicon and document table
            self.lexicon = Lexicon()
            self.document_table = DocumentTable()

            with tqdm(total=total_chunks) as pbar:
                for chunk in chunks:
                    # Process each chunk (it will be a DataFrame)
                    chunk_index = self.process_chunk(chunk)

                    # Update lexicon and document table for each document
                    for _, row in chunk.iterrows():
                        doc_id = row['index']
                        text = row['text']  # Assuming 'text' is preprocessed or tokenized
                        length = len(text.split())  # Length of the document in terms
                        offset = chunk.index[chunk.index == _].tolist()[0]  # Offset of the document in the chunk

                        # Add to Document Table
                        self.document_table.add_document(doc_id, length, offset)

                        # Assuming terms are already extracted from text by process_chunk
                        terms = row['tokens']  # 'tokens' column from the processed chunk
                        for term in terms:
                            self.lexicon.add_term(term, position=offset, term_frequency=terms.count(term))

                    # Write the partial compressed index to a file
                    chunk_index_path = f'Compressed_Index_{len(partial_indices_paths) + 1}.vb'
                    chunk_index.write_index_compressed_to_file(chunk_index_path)
                    partial_indices_paths.append(chunk_index_path)

                    # Clean up memory after processing the chunk
                    del chunk
                    del chunk_index
                    gc.collect()
                    pbar.update(1)

                return partial_indices_paths

        except Exception as e:
            print(f"Error building partial indices: {str(e)}")
            raise

    def build_full_index(self) -> None:
        """Build the full inverted index by merging partial indices."""
        try:
            partial_indices_paths = self.build_partial_indices()

            print("Merging indices...")
            # Merge all partial indices if needed
            self.compressed_inverted_index = self.merger.merge_multiple_compressed_indices(partial_indices_paths)

            total_terms = len(self.compressed_inverted_index.get_terms())
            print(f"Index built successfully with {total_terms} unique terms.")

            # Optionally write the lexicon and document table after merging indices
            self.lexicon.write_to_file("final_lexicon.txt")
            self.document_table.write_to_file("final_document_table.txt")

            # Write the final compressed inverted index
            self.compressed_inverted_index.write_compressed_index_to_file("InvertedIndex")

            # Optionally, delete intermediate files
            self._delete_partial_indices(partial_indices_paths)

        except Exception as e:
            print(f"Error building full index: {str(e)}")
            raise

    @staticmethod
    def _delete_partial_indices(partial_indices_paths: List[str]) -> None:
        """Delete the intermediate index files from disk."""
        for path in partial_indices_paths:
            os.remove(path)
            print(f"Deleted intermediate index file: {path}")

    def build_partial_index(self) -> None:
        """Build a partial inverted index using a sample of documents."""
        try:
            # Sample a subset of documents for partial indexing
            sample_df = self.collection_loader.sample_lines(10000)
            partial_inverted_index = (
                self.compressed_inverted_index.load_compressed_index_to_memory(self.build_partial_indices(sample_df)[0]))

            self.compressed_inverted_index = partial_inverted_index

            print(f"Partial index built with {len(partial_inverted_index.get_terms())} unique terms.")

            # Write the lexicon and document table for the partial index
            self.lexicon.write_to_file("partial_lexicon.txt")
            self.document_table.write_to_file("partial_document_table.txt")

            # Optionally, clean up any intermediate files if needed
            # self._delete_partial_indices(partial_indices_paths)

        except Exception as e:
            print(f"Error building partial index: {str(e)}")
            raise

    def get_index(self) -> CompressedInvertedIndex:
        """Get the built inverted index."""
        return self.compressed_inverted_index

    def get_lexicon(self) -> Lexicon:
        """Get the built inverted index."""
        return self.lexicon

    def get_document_table(self) -> DocumentTable:
        """Get the built inverted index."""
        return self.document_table