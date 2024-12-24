import os
import time

from DocumentTable.DocumentTable import DocumentTable
from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from Lexicon.Lexicon import Lexicon
from Query.QueryParser import QueryParser
from Query.QueryProcessor import QueryProcessor
from Utils.Preprocessing import Preprocessing
from Utils.config import RESOURCES_PATH


def load_resources():
    """
    Loads and returns all required search resources.
    """
    print("Loading resources...")
    query_parser = QueryParser(Preprocessing())  # Using the Preprocessing class here
    lexicon = Lexicon.load_from_file(os.path.join(RESOURCES_PATH, "Lexicon"))
    document_table = DocumentTable.load_from_file(os.path.join(RESOURCES_PATH, "DocumentTable"))
    inverted_index = CompressedInvertedIndex.load_compressed_index_to_memory(
        os.path.join(RESOURCES_PATH, "InvertedIndex"))
    query_processor = QueryProcessor(query_parser, lexicon, document_table, inverted_index)
    print("Resources loaded successfully.")
    return query_processor


def process_query(query_processor, query: str, query_type: str, method: str) -> None:
    """
    Processes the input query using the specified method and prints the results.

    Args:
        query_processor: The QueryProcessor instance to use
        query(str): The user submitted query.
        query_type(str): "1" or "2", corresponding to conjunctive and disjunctive.
        method(str): "1" or "2", corresponding to TFIDF or BM25 score calculation.
    """
    start_time = time.time()

    # Ensure valid method selection
    if method not in ["tfidf", "bm25"]:
        print("Invalid method selection. Please choose 'tfidf' or 'bm25'.")
        return

    # Process query and get ranked documents
    ranked_docs = query_processor.process_query(query, query_type, method)

    end_time = time.time()
    print(f"Query processed in {end_time - start_time:.2f} seconds.")

    # Display ranked documents
    if ranked_docs:
        print("\nRanked documents:")
        for doc_id, score in ranked_docs.items():
            print(f"Document ID: {doc_id}, Score: {score}")
    else:
        print("No results found.")


def main():
    """
    Runs the CLI loop where users can input queries.
    """
    query_processor = load_resources()

    while True:
        query = input("\nEnter query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting the search CLI.")
            break

        if not query:
            print("Query cannot be empty.")
            continue

        # Get the query type (1 for conjunctive, 2 for disjunctive)
        query_type_input = input("Enter query type (1 for conjunctive, 2 for disjunctive): ").strip()
        if query_type_input == "1":
            query_type = "conjunctive"
        elif query_type_input == "2":
            query_type = "disjunctive"
        else:
            print("Invalid query type. Please enter 1 for conjunctive or 2 for disjunctive.")
            continue

        try:
            method_input = int(input("Enter method (1 for TF-IDF, 2 for BM25): "))
            method_mapping = {1: "tfidf", 2: "bm25"}
            if method_input not in method_mapping:
                print("Invalid method selection. Please choose 1 for tfidf or 2 for bm25.")
                continue
            method = method_mapping[method_input]

            # Process and display the results
            process_query(query_processor, query, query_type, method)

        except ValueError:
            print("Invalid input. Please enter 1 for TF-IDF or 2 for BM25.")


if __name__ == "__main__":
    main()