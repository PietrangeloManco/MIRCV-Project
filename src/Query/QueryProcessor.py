from typing import List, Dict
from collections import defaultdict

from DocumentTable.DocumentTable import DocumentTable
from InvertedIndex.CompressedInvertedIndex import CompressedInvertedIndex
from InvertedIndex.Posting import Posting
from Lexicon.Lexicon import Lexicon
from Query.QueryParser import QueryParser
from Query.Scoring import Scoring


class QueryProcessor:
    def __init__(self,
                 query_parser: QueryParser,
                 lexicon: Lexicon,
                 document_table: DocumentTable,
                 inverted_index: CompressedInvertedIndex):
        self.query_parser = query_parser
        self.lexicon = lexicon
        self.document_table = document_table
        self.inverted_index = inverted_index
        self.scoring = Scoring(self.lexicon, self.document_table)

    def process_query(self, query: str, query_type: str = "conjunctive", method: str = "tfidf",
                      max_results: int = 10) -> Dict[int, float]:
        query_terms = self.query_parser.parse(query)
        if not query_terms:
            return {}

        # Get postings lists for each term with their associated term
        term_postings = self.get_term_postings(query_terms)

        # Execute query based on type
        if query_type == "conjunctive":
            matching_docs = self.execute_conjunctive_query(term_postings)
        elif query_type == "disjunctive":
            matching_docs = self.execute_disjunctive_query(term_postings)
        else:
            raise ValueError("Invalid query type. Choose 'conjunctive' or 'disjunctive'.")

        # Rank documents based on the chosen scoring method
        ranked_documents = self.rank_documents(matching_docs, method)

        # Return the top 'max_results' documents
        return dict(list(ranked_documents.items())[:max_results])

    def get_term_postings(self, terms: List[str]) -> Dict[str, List[Posting]]:
        """Get postings for each term while maintaining term association."""
        return {
            term: list(self.inverted_index.get_uncompressed_postings(term))
            for term in terms
        }

    @staticmethod
    def execute_conjunctive_query(term_postings: Dict[str, List[Posting]]) -> Dict[str, Dict[int, Posting]]:
        if not term_postings:
            return {}

        # Get the shortest posting list first to minimize intersection operations
        sorted_terms = sorted(term_postings.items(), key=lambda x: len(x[1]))
        first_term, first_postings = sorted_terms[0]

        # Initialize with the shortest posting list
        matching_doc_ids = {posting.doc_id for posting in first_postings}

        # Intersect with remaining posting lists
        for term, postings in sorted_terms[1:]:
            posting_doc_ids = {posting.doc_id for posting in postings}
            matching_doc_ids &= posting_doc_ids
            if not matching_doc_ids:  # Early termination if no matches
                return {}

        # Create result dictionary only for matching documents
        return {
            term: {
                posting.doc_id: posting
                for posting in postings
                if posting.doc_id in matching_doc_ids
            }
            for term, postings in term_postings.items()
        }

    @staticmethod
    def execute_disjunctive_query(term_postings: Dict[str, List[Posting]]) -> Dict[str, Dict[int, Posting]]:
        """Execute disjunctive query returning matching documents with their postings per term."""
        # Create a map of term -> {doc_id -> posting} for all documents
        return {
            term: {posting.doc_id: posting for posting in postings}
            for term, postings in term_postings.items()
        }

    def rank_documents(self, term_postings: Dict[str, Dict[int, Posting]], method: str) -> Dict[int, float]:
        """Rank documents using the correct posting for each term-document pair with query coverage boosting."""
        scores = defaultdict(float)

        # Get all unique document IDs
        all_doc_ids = {
            doc_id
            for postings_map in term_postings.values()
            for doc_id in postings_map.keys()
        }

        # For each document, compute its score
        for doc_id in all_doc_ids:
            doc_score = 0
            matched_terms = 0
            for term, postings_map in term_postings.items():
                if doc_id in postings_map:
                    # Only compute score if the term appears in the document
                    posting = postings_map[doc_id]
                    doc_score += self.scoring.compute_score(term, doc_id, posting.payload, method)
                    matched_terms += 1

            scores[doc_id] = doc_score

        # Rank the documents by their score in descending order
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
