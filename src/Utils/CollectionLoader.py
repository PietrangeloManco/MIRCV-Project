from typing import List
import pandas as pd
from pandas import DataFrame

# Auxiliary class to read the collection out of the tsv file
class CollectionLoader:
    def __init__(self, file_path: str = 'C:\\Users\pietr\OneDrive\Documenti\GitHub\MIRCV-Project\Files\collection.tsv',
                 chunk_size: int = 100000, column_names: List[str] = None):
        if column_names is None:
            column_names = ['index', 'text']
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.column_names = column_names

    # Method to extract the full text
    def process_chunks(self) -> DataFrame:
        chunks = []
        for chunk in pd.read_csv(self.file_path, sep='\t', chunksize=self.chunk_size, names=self.column_names, header=0):
            chunks.append(chunk)
        df = pd.concat(chunks, axis=0)
        return df

    # Method to take just some lines
    def sample_lines(self, num_lines: int = 10) -> DataFrame:
        sampled_lines = []
        total_sampled = 0
        for chunk in pd.read_csv(self.file_path, sep='\t', chunksize=self.chunk_size, names=self.column_names, header=0):
            if total_sampled < num_lines:
                sampled_chunk = chunk.sample(n=min(num_lines - total_sampled, len(chunk)))
                sampled_lines.append(sampled_chunk)
                total_sampled += len(sampled_chunk)
            else:
                break
        sample_df = pd.concat(sampled_lines, axis=0)
        print(sample_df.columns)
        return sample_df