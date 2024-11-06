from collections import defaultdict
from typing import Any, List, Iterator
from src.InvertedIndex.Posting import Posting

# Inverted Index structure class
class InvertedIndex:

    def __init__(self):
        self._index = defaultdict(list)

    # Add a posting to the inverted index
    def add_posting(self, term: str, doc_id: int, payload: Any = None) -> None:
        """Adds a document to the posting list of a term."""
        self._index[term].append(Posting(doc_id, payload))

    def add_postings(self, term: str, postings: Iterator[Posting]) -> None:
        """Adds a collection of postings for a given term."""
        self._index[term].extend(postings)

    # Get the posting for a given term from the inverted index
    def get_postings(self, term: str) -> List[Posting]:
        """Fetches the posting list for a given term."""
        return self._index.get(term)

    # Get the list of unique terms in the inverted index
    def get_terms(self) -> List[str]:
        """Returns all unique terms in the index."""
        return list(self._index.keys())

    # Write the inverted index to a file
    def write_to_file(self, filename_index: str) -> None:
        """Saves the index to a textfile."""
        with open(filename_index, "w") as f:
            for term, postings in self._index.items():
                f.write(term)
                for posting in postings:
                    f.write(f" {posting.doc_id}")
                    if posting.payload:
                        f.write(f":{str(posting.payload)}")
                f.write("\n")

    @staticmethod
    def load_from_file(filename_index: str) -> 'InvertedIndex':
        """Loads an inverted index from a textfile."""
        index = InvertedIndex()
        with open(filename_index, "r") as f:
            for line in f:
                terms = line.strip().split()
                term = terms[0]
                for term_data in terms[1:]:
                    doc_id, *payload = term_data.split(":")
                    payload = ":".join(payload) if payload else None
                    index.add_posting(term, int(doc_id), payload)
        return index