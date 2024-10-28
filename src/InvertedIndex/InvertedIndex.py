from collections import defaultdict
from typing import Any, List

from src.InvertedIndex.Posting import Posting


class InvertedIndex:

    def __init__(self):
        self._index = defaultdict(list)

    def add_posting(self, term: str, doc_id: int, payload: Any = None) -> None:
        """Adds a document to the posting list of a term."""
        self._index[term].append(Posting(doc_id, payload))

    def get_postings(self, term: str) -> List[Posting]:
        """Fetches the posting list for a given term."""
        return self._index.get(term)

    def get_terms(self) -> List[str]:
        """Returns all unique terms in the index."""
        return self._index.keys()

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