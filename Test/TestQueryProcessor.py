import unittest

from DocumentTable.DocumentTable import DocumentTable
from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from Lexicon.Lexicon import Lexicon
from Query.QueryParser import QueryParser
from Query.QueryProcessor import QueryProcessor
from Utils.CollectionLoader import CollectionLoader
from Utils.Preprocessing import Preprocessing

class TestQueryProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This method is called once before any test in the class is executed.
        It's used to create common resources needed for the tests.
        """
        # Assuming instances are being created from files or mock data
        cls.query_parser = QueryParser(Preprocessing())  # Replace with actual initialization
        cls.lexicon = Lexicon.load_from_file("Lexicon")
        cls.document_table = DocumentTable.load_from_file("DocumentTable")
        cls.inverted_index = CompressedInvertedIndex.load_compressed_index_to_memory("InvertedIndex")
        cls.query_processor = QueryProcessor(
            cls.query_parser, cls.lexicon, cls.document_table, cls.inverted_index
        )

    def test_conjunctive_query_tfidf(self):
        query = "data mining"
        query_type = "conjunctive"
        method = "tfidf"

        # Call the method with a sample query and check if it's processed correctly
        ranked_docs = self.query_processor.process_query(query, query_type, method)

        doc_ids = list(ranked_docs.keys())  # Get the list of doc IDs from the ranked_docs

        # Call the get_documents_by_ids method and print the documents
        documents = CollectionLoader().get_documents_by_ids(doc_ids)

        # Print the documents along with their scores
        for doc_id, document in zip(doc_ids, documents):
            print(f"Document ID: {doc_id}")
            print(f"Document Text: {document}")
            print("-" * 40)  # Separator for readability

        # Add checks based on your expected outcomes
        self.assertIsInstance(ranked_docs, dict)
        self.assertGreater(len(ranked_docs), 0)
        # Check if the documents are sorted by score in descending order
        scores = list(ranked_docs.values())
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_disjunctive_query_tfidf(self):
        query = "data OR mining"
        query_type = "disjunctive"
        method = "tfidf"

        # Call the method with a sample query and check if it's processed correctly
        ranked_docs = self.query_processor.process_query(query, query_type, method)

        doc_ids = list(ranked_docs.keys())  # Get the list of doc IDs from the ranked_docs

        # Call the get_documents_by_ids method and print the documents
        documents = CollectionLoader().get_documents_by_ids(doc_ids)

        # Print the documents along with their scores
        for doc_id, document in zip(doc_ids, documents):
            print(f"Document ID: {doc_id}")
            print(f"Document Text: {document}")
            print("-" * 40)  # Separator for readability

        # Add checks based on your expected outcomes
        self.assertIsInstance(ranked_docs, dict)
        self.assertGreater(len(ranked_docs), 0)
        # Check if the documents are sorted by score in descending order
        scores = list(ranked_docs.values())
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_conjunctive_query_bm25(self):
        query = "machine learning"
        query_type = "conjunctive"
        method = "bm25"

        # Call the method with a sample query and check if it's processed correctly
        ranked_docs = self.query_processor.process_query(query, query_type, method)

        doc_ids = list(ranked_docs.keys())  # Get the list of doc IDs from the ranked_docs

        # Call the get_documents_by_ids method and print the documents
        documents = CollectionLoader().get_documents_by_ids(doc_ids)

        # Print the documents along with their scores
        for doc_id, document in zip(doc_ids, documents):
            print(f"Document ID: {doc_id}")
            print(f"Document Text: {document}")
            print("-" * 40)  # Separator for readability

        # Add checks based on your expected outcomes
        self.assertIsInstance(ranked_docs, dict)
        self.assertGreater(len(ranked_docs), 0)
        # Check if the documents are sorted by score in descending order
        scores = list(ranked_docs.values())
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_disjunctive_query_bm25(self):
        query = "data OR mining"
        query_type = "disjunctive"
        method = "bm25"

        # Call the method with a sample query and check if it's processed correctly
        ranked_docs = self.query_processor.process_query(query, query_type, method)

        doc_ids = list(ranked_docs.keys())  # Get the list of doc IDs from the ranked_docs

        # Call the get_documents_by_ids method and print the documents
        documents = CollectionLoader().get_documents_by_ids(doc_ids)

        # Print the documents along with their scores
        for doc_id, document in zip(doc_ids, documents):
            print(f"Document ID: {doc_id}")
            print(f"Document Text: {document}")
            print("-" * 40)  # Separator for readability
        # Add checks based on your expected outcomes
        self.assertIsInstance(ranked_docs, dict)
        self.assertGreater(len(ranked_docs), 0)
        # Check if the documents are sorted by score in descending order
        scores = list(ranked_docs.values())
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query(self):
        query = ""
        query_type = "conjunctive"
        method = "tfidf"

        ranked_docs = self.query_processor.process_query(query, query_type, method)

        # Assert that the result is an empty dictionary for an empty query
        self.assertEqual(ranked_docs, {})

    def test_invalid_query_type(self):
        query = "data mining"
        query_type = "invalid_query_type"
        method = "tfidf"

        # Assert that calling with an invalid query type raises a ValueError
        with self.assertRaises(ValueError):
            self.query_processor.process_query(query, query_type, method)

    def test_invalid_method(self):
        query = "data mining"
        query_type = "conjunctive"
        method = "invalid_method"

        # Assert that calling with an invalid scoring method raises a ValueError
        with self.assertRaises(ValueError):
            self.query_processor.process_query(query, query_type, method)

    def test_ranking_order(self):
        query = "data mining"
        query_type = "conjunctive"
        method = "tfidf"

        ranked_docs = self.query_processor.process_query(query, query_type, method)

        # Check that the documents are ordered by score in descending order
        scores = list(ranked_docs.values())
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
