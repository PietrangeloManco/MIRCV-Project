from typing import List, Dict
from collections import defaultdict

from DocumentTable.DocumentTable import DocumentTable
from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from Lexicon.Lexicon import Lexicon
from Query.QueryParser import QueryParser
from Query.Scoring import Scoring


class QueryProcessor:
    def __init__(self,
                 query_parser: QueryParser,
                 lexicon: Lexicon,
                 document_table: DocumentTable,
                 inverted_index: CompressedInvertedIndex):
        """
        Initializes the QueryProcessor.
        :param query_parser: An instance of the QueryParser class for parsing queries.
        :param lexicon: An instance of the Lexicon class to retrieve term frequency and document frequency.
        :param document_table: An instance of the DocumentTable class to retrieve document lengths.
        :param inverted_index: An inverted index mapping terms to document IDs.
        """
        self.query_parser = query_parser
        self.lexicon = lexicon
        self.document_table = document_table
        self.inverted_index = inverted_index
        self.scoring = Scoring(self.lexicon, self.document_table)

    def process_query(self, query: str, query_type: str = "conjunctive", method: str = "tfidf", max_results: int = 10) -> Dict[int, float]:
        """
        Processes a query and returns the ranked documents.
        :param query: The input query string.
        :param query_type: The type of query: 'conjunctive' or 'disjunctive'.
        :param method: The scoring method: 'tfidf' or 'bm25'.
        :param max_results: The maximum number of relevant documents to return.
        :return: A dictionary where keys are document IDs and values are their scores.
        """
        # Step 1: Parse the query into terms
        query_terms = self.query_parser.parse(query)
        if not query_terms:
            return {}

        # Step 2: Execute the query (conjunctive or disjunctive)
        if query_type == "conjunctive":
            postings = self.execute_conjunctive_query(query_terms)
        elif query_type == "disjunctive":
            postings = self.execute_disjunctive_query(query_terms)
        else:
            raise ValueError("Invalid query type. Choose 'conjunctive' or 'disjunctive'.")

        # Step 3: Rank the documents based on the chosen scoring method
        ranked_documents = self.rank_documents(postings, query_terms, method)

        # Step 4: Limit the results to the top 'max_results' documents
        return dict(list(ranked_documents.items())[:max_results+1])

    def execute_conjunctive_query(self, terms: List[str]) -> List[int]:
        """
        Executes a conjunctive query (AND operation) to find documents containing all terms.
        :param terms: The list of query terms.
        :return: A list of document IDs that contain all query terms.
        """
        # Start with the postings list of the first term (extract doc_ids)
        postings = {posting.doc_id for posting in self.inverted_index.get_uncompressed_postings(terms[0])}

        # Intersect with the postings lists of the other terms
        for term in terms[1:]:
            postings &= {posting.doc_id for posting in self.inverted_index.get_uncompressed_postings(term)}

        return list(postings)

    def execute_disjunctive_query(self, terms: List[str]) -> List[int]:
        """
        Executes a disjunctive query (OR operation) to find documents containing any of the terms.
        :param terms: The list of query terms.
        :return: A list of document IDs that contain at least one query term.
        """
        postings = set()
        for term in terms:
            postings |= {posting.doc_id for posting in self.inverted_index.get_uncompressed_postings(term)}

        return list(postings)

    def rank_documents(self, postings: List[int], query_terms: List[str], method: str) -> Dict[int, float]:
        """
        Ranks the documents based on the specified scoring method.
        :param postings: The list of document IDs to rank.
        :param query_terms: The query terms to compute the score for.
        :param method: The scoring method to use ('tfidf' or 'bm25').
        :return: A dictionary with document IDs as keys and their scores as values.
        """
        scores = defaultdict(float)

        for doc_id in postings:
            for term in query_terms:
                # Compute the score for each term and document using the specified method
                score = self.scoring.compute_score(term, doc_id, method)
                scores[doc_id] += score

        # Sort documents by score in descending order
        ranked_scores = {doc_id: score for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)}

        return ranked_scores
