from typing import List, Iterator
from pandas import DataFrame
import random
import pandas as pd
import io
import gzip

class CollectionLoader:
    def __init__(self,
                 file_path: str = 'C:\\Users\\pietr\\OneDrive\\Documenti\\GitHub\\MIRCV-Project\\Files\\collection.tar.gz',
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

    def process_chunks(self, chunk_size: int = None) -> Iterator[DataFrame]:
        """
        Process collection in chunks, yielding DataFrames

        Args:
            chunk_size: Optional override for chunk size

        Yields:
            DataFrame: Chunk of documents
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        with gzip.open(self.file_path, 'rt', encoding='utf-8') as file:
            next(file)  # Skip header
            chunk = []

            for line in file:
                try:
                    # Split line into columns and strip whitespace
                    columns = line.strip().split('\t')
                    if len(columns) == len(self.column_names):
                        chunk.append(columns)

                    if len(chunk) >= chunk_size:
                        df = pd.DataFrame(chunk, columns=self.column_names)
                        # Convert index column to integer
                        df['index'] = pd.to_numeric(df['index'], errors='coerce')
                        # Drop rows with invalid indices
                        df = df.dropna(subset=['index'])
                        df['index'] = df['index'].astype(int)
                        yield df
                        chunk = []
                except Exception as e:
                    print(f"Warning: Skipping malformed line: {str(e)}")
                    continue

            # Handle last chunk if it exists
            if chunk:
                df = pd.DataFrame(chunk, columns=self.column_names)
                df['index'] = pd.to_numeric(df['index'], errors='coerce')
                df = df.dropna(subset=['index'])
                df['index'] = df['index'].astype(int)
                yield df

    def sample_lines(self, num_lines: int = 10) -> DataFrame:
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

        return sample_df
