import sys
from typing import List, Iterator

from pandas import DataFrame
import random
import pandas as pd
import io
import gzip

class CollectionLoader:
    def __init__(self,
                 file_path: str = "C:\\Users\\pietr\\OneDrive\\Documenti\\GitHub\\MIRCV-Project\\Files\\collection.tar.gz",
                 chunk_size: int = 500000,
                 column_names: List[str] = None):
        if column_names is None:
            column_names = ['index', 'text']
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.column_names = column_names
        self._total_docs = None  # Cache for total documents count

    def get_total_docs(self) -> int:
        """Get total number of documents in collection"""
        if self._total_docs is None:
            print("Computing documents number...")
            # Count lines efficiently without loading the file
            with gzip.open(self.file_path, 'rt', encoding='utf-8') as file:
                next(file)  # Skip header
                self._total_docs = sum(1 for _ in file)
        return self._total_docs

    def process_single_chunk(self, start: int, chunk_size: int) -> pd.DataFrame:
        """
        Process a single chunk of the collection starting from a given line.

        Args:
            start: Line number to start reading from.
            chunk_size: Number of lines to read in the chunk.

        Returns:
            DataFrame: DataFrame containing the processed chunk.
        """
        chunk = []

        with gzip.open(self.file_path, 'rt', encoding='utf-8') as file:
            next(file)  # Skip header
            for _ in range(start):
                next(file)  # Skip lines until the start point

            for _ in range(chunk_size):
                line = file.readline()
                if not line:
                    break  # End of file reached

                columns = line.strip().split('\t')
                if len(columns) == len(self.column_names):
                    chunk.append(columns)

        if chunk:
            df = pd.DataFrame(chunk, columns=self.column_names)
            df['index'] = pd.to_numeric(df['index'], errors='coerce')
            df = df.dropna(subset=['index'])
            df['index'] = df['index'].astype(int)
            return df

        return pd.DataFrame(columns=self.column_names)

    def process_chunks(self, chunk_size: int = None) -> Iterator[pd.DataFrame]:
        """
        Process the entire collection in chunks, yielding DataFrames.

        Args:
            chunk_size: Optional override for chunk size.

        Yields:
            DataFrame: Chunk of documents.
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        total_lines = self.get_total_docs()  # Implement this method if needed
        start = 0

        while start < total_lines:
            yield self.process_single_chunk(start, chunk_size)
            start += chunk_size

    def sample_lines(self, num_lines: int = 10) -> pd.DataFrame:
        """
        Sample random lines from collection using reservoir sampling

        Args:
            num_lines: Number of lines to sample

        Returns:
            DataFrame: Sampled documents
        """
        self._total_docs = num_lines
        with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
            next(f)  # Skip header
            reservoir = []

            for i, line in enumerate(f, 1):
                if len(reservoir) < num_lines:
                    reservoir.append(line)
                else:
                    j = random.randint(0, i)
                    if j < num_lines:
                        reservoir[j] = line

        # Process sampled lines
        sample_df = pd.read_csv(
            io.StringIO(''.join(reservoir)),
            sep='\t',
            names=self.column_names
        )

        # Convert index to integer
        sample_df['index'] = pd.to_numeric(sample_df['index'], errors='coerce')
        sample_df = sample_df.dropna(subset=['index'])
        sample_df['index'] = sample_df['index'].astype(int)

        # Sort the DataFrame by the 'index' column (document IDs)
        sample_df = sample_df.sort_values(by='index').reset_index(drop=True)

        return sample_df

    def get_documents_by_ids(self, docids: List[int]) -> List[str]:
        """
        Retrieves the documents corresponding to the given list of docids.

        Args:
            docids: List of document IDs to retrieve.

        Returns:
            List of str: A list of document texts.
        """
        documents = []
        # Iterate over chunks of the collection
        for chunk in self.process_chunks(sys.maxsize):
            # Filter the chunk for matching docids
            matching_docs = chunk[chunk['index'].isin(docids)]

            # If matching documents exist in this chunk, extract the text
            if not matching_docs.empty:
                documents.extend(matching_docs['text'].tolist())

            # If we've found all documents, we can stop early
            if len(documents) == len(docids):
                break

        # Return the list of document texts (or a message if not found)
        return documents if documents else ["Not existing document" for _ in docids]
