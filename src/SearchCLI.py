import time
from Query.QueryProcessor import QueryProcessor
from DocumentTable.DocumentTable import DocumentTable
from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from Lexicon.Lexicon import Lexicon
from Query.QueryParser import QueryParser
from Utils.Preprocessing import Preprocessing

class SearchCLI:
    def __init__(self):
        """
        Initializes the CLI by loading the required structures into memory.
        """
        resources_path = "C:\\Users\\pietr\\OneDrive\\Documenti\\GitHub\\MIRCV-Project\\Files\\"
        print("Loading resources...")
        self.query_parser = QueryParser(Preprocessing())  # Using the Preprocessing class here
        self.lexicon = Lexicon.load_from_file(resources_path + "Lexicon")
        self.document_table = DocumentTable.load_from_file(resources_path + "DocumentTable")
        self.inverted_index = CompressedInvertedIndex.load_compressed_index_to_memory(resources_path + "InvertedIndex")
        self.query_processor = QueryProcessor(
            self.query_parser, self.lexicon, self.document_table, self.inverted_index
        )
        print("Resources loaded successfully.")

    def process_query(self, query, query_type, method):
        """
        Processes the input query using the specified method and prints the results.
        """
        start_time = time.time()

        # Ensure valid method selection
        if method not in ["tfidf", "bm25"]:
            print("Invalid method selection. Please choose 'tfidf' or 'bm25'.")
            return

        # Process query and get ranked documents
        ranked_docs = self.query_processor.process_query(query, query_type, method)

        end_time = time.time()
        print(f"Query processed in {end_time - start_time:.2f} seconds.")

        # Display ranked documents
        if ranked_docs:
            print("\nRanked documents:")
            for doc_id, score in ranked_docs.items():
                print(f"Document ID: {doc_id}, Score: {score}")
        else:
            print("No results found.")

    def run(self):
        """
        Runs the CLI loop where users can input queries.
        """
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
                self.process_query(query, query_type, method)

            except ValueError:
                print("Invalid input. Please enter 1 for TF-IDF or 2 for BM25.")

if __name__ == "__main__":
    cli = SearchCLI()
    cli.run()
