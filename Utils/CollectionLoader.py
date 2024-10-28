import pandas as pd


class CollectionLoader:
    def __init__(self, file_path, chunk_size=100000):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def process_chunks(self):
        chunks = []
        for chunk in pd.read_csv(self.file_path, sep='\t', chunksize=self.chunk_size):
            # Process each chunk as needed
            # For example, you can filter, transform, or analyze the data here
            chunks.append(chunk)

        # Concatenate all chunks into a single DataFrame
        df = pd.concat(chunks, axis=0)
        return df

    def sample_lines(self, num_lines=10):
        # Read the file in chunks and sample lines
        sampled_lines = []
        total_sampled = 0
        for chunk in pd.read_csv(self.file_path, sep='\t', chunksize=self.chunk_size):
            if total_sampled < num_lines:
                sampled_chunk = chunk.sample(n=min(num_lines - total_sampled, len(chunk)))
                sampled_lines.append(sampled_chunk)
                total_sampled += len(sampled_chunk)
            else:
                break

        # Concatenate sampled lines into a single DataFrame
        sample_df = pd.concat(sampled_lines, axis=0)
        return sample_df

        # Concatenate sampled lines into a single DataFrame
        sample_df = pd.concat(sampled_lines, axis=0)
        return sample_df

    def save_processed_data(self, output_path):
        df = self.process_chunks()
        df.to_csv(output_path, sep='\t', index=False)