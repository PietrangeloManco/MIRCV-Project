from multiprocessing import get_context, cpu_count
import pandas as pd

def process_chunk(chunk):
    # Dummy processing function
    return len(chunk)

def main():
    ctx = get_context("spawn")
    n_cores = cpu_count() - 1
    chunk_size = 1000

    # Create a dummy DataFrame
    data = {'text': ['doc1', 'doc2', 'doc3'], 'index': [1, 2, 3]}
    df = pd.DataFrame(data)

    # Split DataFrame into chunks
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    with ctx.Pool(processes=n_cores) as pool:
        results = pool.map(process_chunk, chunks)
        print(results)

if __name__ == "__main__":
    main()

