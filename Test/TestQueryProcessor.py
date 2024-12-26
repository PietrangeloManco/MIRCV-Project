import os
import time
import unittest

from DocumentTable.DocumentTable import DocumentTable
from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from Lexicon.Lexicon import Lexicon
from Query.QueryParser import QueryParser
from Query.QueryProcessor import QueryProcessor
from Utils.CollectionLoader import CollectionLoader
from Utils.Preprocessing import Preprocessing
from Utils.config import RESOURCES_PATH


class TestQueryProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Class resources."""
        cls.query_parser = QueryParser(Preprocessing())
        cls.lexicon = Lexicon.load_from_file(os.path.join(RESOURCES_PATH, "Lexicon"))
        cls.document_table = DocumentTable.load_from_file(os.path.join(RESOURCES_PATH, "DocumentTable"))
        cls.inverted_index = CompressedInvertedIndex.load_compressed_index_to_memory(
            os.path.join(RESOURCES_PATH, "InvertedIndex"))
        cls.query_processor = QueryProcessor(
            cls.query_parser, cls.lexicon, cls.document_table, cls.inverted_index
        )

    def test_conjunctive_query_tfidf(self):
        """Test of the processing of a conjunctive query with tfidf scoring."""
        query = "information retrieval"
        query_type = "conjunctive"
        method = "tfidf"

        start_time = time.time()
        ranked_docs = self.query_processor.process_query(query, query_type, method)
        end_time = time.time()

        doc_ids = list(ranked_docs.keys())

        print(f"Retrieved in {end_time - start_time}")
        # Print the document ids along with their scores
        for doc_id in zip(doc_ids):
            print(f"Document ID: {doc_id}")

        # Checks
        self.assertIsInstance(ranked_docs, dict)
        self.assertGreaterEqual(len(ranked_docs), 0)
        # Check if the documents are sorted by score in descending order
        scores = list(ranked_docs.values())
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_disjunctive_query_tfidf(self):
        """Test of the processing of a disjunctive query with tfidf scoring."""
        query = "information retrieval"
        query_type = "disjunctive"
        method = "bm25"

        start_time = time.time()

        ranked_docs = self.query_processor.process_query(query, query_type, method)
        end_time = time.time()

        doc_ids = list(ranked_docs.keys())

        print(f"Retrieved in {end_time - start_time}")
        # Print the document ids along with their scores
        for doc_id in zip(doc_ids):
            print(f"Document ID: {doc_id}")

        self.assertIsInstance(ranked_docs, dict)
        self.assertGreater(len(ranked_docs), 0)
        # Check if the documents are sorted by score in descending order
        scores = list(ranked_docs.values())
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_conjunctive_query_bm25(self):
        """Test of the processing of a conjunctive query with bm25 scoring."""
        query = "information retrieval"
        query_type = "conjunctive"
        method = "bm25"

        start_time = time.time()

        ranked_docs = self.query_processor.process_query(query, query_type, method)
        end_time = time.time()

        doc_ids = list(ranked_docs.keys())

        print(f"Retrieved in {end_time - start_time}")
        # Print the document ids along with their scores
        for doc_id in zip(doc_ids):
            print(f"Document ID: {doc_id}")

        self.assertIsInstance(ranked_docs, dict)
        self.assertGreater(len(ranked_docs), 0)
        # Check if the documents are sorted by score in descending order
        scores = list(ranked_docs.values())
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_disjunctive_query_bm25(self):
        """Test of the processing of a disjunctive query with bm25 scoring."""
        query = "information retrieval"
        query_type = "disjunctive"
        method = "bm25"

        start_time = time.time()

        ranked_docs = self.query_processor.process_query(query, query_type, method)
        end_time = time.time()

        doc_ids = list(ranked_docs.keys())

        print(f"Retrieved in {end_time - start_time}")
        # Print the document ids along with their scores
        for doc_id in zip(doc_ids):
            print(f"Document ID: {doc_id}")

        self.assertIsInstance(ranked_docs, dict)
        self.assertGreater(len(ranked_docs), 0)
        # Check if the documents are sorted by score in descending order
        scores = list(ranked_docs.values())
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_fully_retrieved_documents(self):
        """This method retrieves documents' text, in order to check that useful documents are
        retrieved. It takes a bit longer because it has to directly access collection."""
        query = "where to eat pizza or pasta in rome"
        query_type = "disjunctive"
        method = "bm25"

        start_time = time.time()

        ranked_docs = self.query_processor.process_query(query, query_type, method)
        end_time = time.time()

        doc_ids = list(ranked_docs.keys())  # Get the list of doc IDs from the ranked_docs
        # Call the get_documents_by_ids method and print the documents
        documents = CollectionLoader().get_documents_by_ids(doc_ids)

        print(f"Retrieved in {end_time - start_time}")
        # Print the documents along with their scores
        for doc_id, document in zip(doc_ids, documents):
            print(f"Document ID: {doc_id}")
            print(f"Document Text: {document}")
            print("-" * 40)  # Separator for readability

        self.assertIsInstance(ranked_docs, dict)
        self.assertGreater(len(ranked_docs), 0)
        # Check if the documents are sorted by score in descending order
        scores = list(ranked_docs.values())
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query(self):
        """Test the response to an empty query with default settings."""
        query = ""
        query_type = "conjunctive"
        method = "tfidf"

        ranked_docs = self.query_processor.process_query(query, query_type, method)

        # Assert that the result is an empty dictionary for an empty query
        self.assertEqual(ranked_docs, {})

    def test_invalid_query_type(self):
        """Test the response to a query with an invalid query type."""
        query = "data mining"
        query_type = "invalid_query_type"
        method = "tfidf"

        # Assert that calling with an invalid query type raises a ValueError
        with self.assertRaises(ValueError):
            self.query_processor.process_query(query, query_type, method)

    def test_invalid_method(self):
        """Test the response to a query with invalid scoring function."""
        query = "data mining"
        query_type = "conjunctive"
        method = "invalid_method"

        # Assert that calling with an invalid scoring method raises a ValueError
        with self.assertRaises(ValueError):
            self.query_processor.process_query(query, query_type, method)


if __name__ == "__main__":
    unittest.main()
