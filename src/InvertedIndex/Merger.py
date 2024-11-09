import concurrent.futures
from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from Utils.CompressionTools import CompressionTools

class Merger:
    def __init__(self):
        """Initialize the Merger class."""
        pass

    @staticmethod
    def _merge_compressed_postings(postings1: bytes, postings2: bytes) -> bytes:
        """Merge two lists of compressed postings."""
        if not postings1:
            return postings2
        if not postings2:
            return postings1

        # Decompress both postings lists
        doc_ids1 = CompressionTools.pfor_delta_decompress(postings1)
        doc_ids2 = CompressionTools.pfor_delta_decompress(postings2)
        merged_doc_ids = sorted(set(doc_ids1 + doc_ids2))
        return CompressionTools.pfor_delta_compress(merged_doc_ids)

    def _merge_two_indices(self, index1: CompressedInvertedIndex, index2: CompressedInvertedIndex) -> CompressedInvertedIndex:
        """Helper function to merge two compressed indices."""
        merged_index = CompressedInvertedIndex()
        all_terms = set(index1.get_terms()).union(index2.get_terms())

        for term in all_terms:
            postings1 = index1.get_compressed_postings(term)
            postings2 = index2.get_compressed_postings(term)
            merged_postings = self._merge_compressed_postings(postings1, postings2)
            if merged_postings:
                merged_index.add_compressed_postings(term, merged_postings)

        return merged_index

    def merge_multiple_compressed_indices(self, index_paths: list[str]) -> CompressedInvertedIndex:
        """Merge an arbitrary number of compressed indices using parallel merging with ProcessPoolExecutor."""
        if not index_paths:
            raise ValueError("The list of index paths is empty.")

        # Load all indices into memory
        indices = [CompressedInvertedIndex.load_compressed_index_to_memory(path) for path in index_paths]

        # Keep merging until there is only one index left
        while len(indices) > 1:
            merged_results = []

            # Use ProcessPoolExecutor for true parallelism
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for i in range(0, len(indices) - 1, 2):
                    # Schedule merging of index pairs
                    futures.append(executor.submit(self._merge_two_indices, indices[i], indices[i + 1]))

                # Collect the merged results
                for future in concurrent.futures.as_completed(futures):
                    merged_results.append(future.result())

            # If there is an odd number of indices, add the last one to the results
            if len(indices) % 2 == 1:
                merged_results.append(indices[-1])

            # Update the indices list with the merged results
            indices = merged_results

        # Return the final merged index
        return indices[0]
