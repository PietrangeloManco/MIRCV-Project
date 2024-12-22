from sklearn.metrics import ndcg_score
from tqdm import tqdm
import numpy as np

from DocumentTable.DocumentTable import DocumentTable
from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from Lexicon.Lexicon import Lexicon
from Query.QueryParser import QueryParser
from Query.QueryProcessor import QueryProcessor
from Utils.Preprocessing import Preprocessing


class EvaluationCLI:
    def __init__(self):
        """
        Initializes the CLI by loading the required structures into memory.
        """
        self.resources_path = "C:\\Users\\pietr\\OneDrive\\Documenti\\GitHub\\MIRCV-Project\\Files\\"
        print("Loading resources...")
        self.query_parser = QueryParser(Preprocessing())
        self.lexicon = Lexicon.load_from_file(self.resources_path + "Lexicon")
        self.document_table = DocumentTable.load_from_file(self.resources_path + "DocumentTable")
        self.inverted_index = CompressedInvertedIndex.load_compressed_index_to_memory(
            self.resources_path + "InvertedIndex")
        self.query_processor = QueryProcessor(
            self.query_parser, self.lexicon, self.document_table, self.inverted_index
        )
        # Add cache for query results
        self.query_results_cache = {}
        print("Resources loaded successfully.")

    @staticmethod
    def compute_ndcg(ranked_docs, query_id, qrels):
        """
        Optimized NDCG computation using numpy arrays.
        """
        judged_docs = {doc_id for q, doc_id in qrels.keys() if q == query_id}
        if not judged_docs:
            return None

        # Convert to numpy arrays for faster computation
        relevance_scores = np.array([
            qrels.get((query_id, doc_id), 0) for doc_id in ranked_docs
        ])

        ideal_relevance_scores = np.array([
            qrels.get((query_id, doc_id), 0) for doc_id in judged_docs
        ])
        ideal_relevance_scores.sort()
        ideal_relevance_scores = ideal_relevance_scores[::-1]  # Reverse for descending order

        # Pad if necessary
        if len(relevance_scores) < len(ideal_relevance_scores):
            relevance_scores = np.pad(
                relevance_scores,
                (0, len(ideal_relevance_scores) - len(relevance_scores))
            )

        # Reshape for ndcg_score
        relevance_scores = relevance_scores.reshape(1, -1)
        ideal_relevance_scores = ideal_relevance_scores.reshape(1, -1)

        try:
            return ndcg_score(ideal_relevance_scores, relevance_scores)
        except ValueError:
            return 0.0

    def process_query(self, query_tuple, query_type, method, qrels):
        """
        Processes the query with caching to avoid redundant computations.
        """
        query_id, query_text = query_tuple
        if isinstance(query_id, str):
            query_id = int(query_id)

        # Create a cache key
        cache_key = (query_id, query_type, method)

        # Check if we have cached results
        if cache_key not in self.query_results_cache:
            ranked_docs = self.query_processor.process_query(query_text, query_type, method)
            self.query_results_cache[cache_key] = ranked_docs
        else:
            ranked_docs = self.query_results_cache[cache_key]

        return self.compute_ndcg(ranked_docs, query_id, qrels)

    def evaluate_all_queries(self, queries, qrels):
        """
        Evaluate all queries with optimized data structures and progress tracking.
        """
        # Pre-process qrels for faster lookup
        query_relevance = {}
        for (query_id, doc_id), relevance in qrels.items():
            if query_id not in query_relevance:
                query_relevance[query_id] = {}
            query_relevance[query_id][doc_id] = relevance

        results = {
            "conjunctive_tfidf": [],
            "conjunctive_bm25": [],
            "disjunctive_tfidf": [],
            "disjunctive_bm25": []
        }

        # Filter out queries without relevance judgments
        valid_queries = [
            query for query in queries
            if query[0] in query_relevance
        ]

        # Process queries with progress bar
        with tqdm(total=len(valid_queries) * 4, desc="Processing queries") as pbar:
            for query_tuple in valid_queries:
                for query_type in ["conjunctive", "disjunctive"]:
                    for method in ["tfidf", "bm25"]:
                        ndcg = self.process_query(query_tuple, query_type, method, qrels)
                        if ndcg is not None:
                            results[f"{query_type}_{method}"].append(ndcg)
                        pbar.update(1)

        return results

    def run(self):
        """
        Runs the evaluation process with improved progress tracking.
        """
        print("Loading qrels and queries...")
        qrels = self.load_qrels(self.resources_path + "2020qrels-pass.txt")
        queries = self.load_queries(self.resources_path + "msmarco-test2020-queries.tsv")

        print("\nEvaluating queries...")
        results = self.evaluate_all_queries(queries, qrels)

        print("\nFinal Results:")
        for combination, ndcg_scores in results.items():
            avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
            print(f"Average NDCG for {combination}: {avg_ndcg:.4f}")

    @staticmethod
    def load_qrels(qrels_file_path):
        """
        Loads the qrels file into a dictionary for quick lookup.
        """
        qrels = {}
        with open(qrels_file_path, 'r') as f:
            for line in f:
                query_id, _, doc_id, relevance = line.split()
                qrels[(int(query_id), int(doc_id))] = int(relevance)
        return qrels

    @staticmethod
    def load_queries(queries_file_path):
        """
        Loads the query file into a list of queries.
        """
        queries = []
        with open(queries_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    queries.append((int(parts[0]), parts[1].strip()))
        return queries


if __name__ == "__main__":
    cli = EvaluationCLI()
    cli.run()