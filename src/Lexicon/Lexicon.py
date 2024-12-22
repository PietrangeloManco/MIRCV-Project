# Lexicon structure class
from typing import List


class Lexicon:
    def __init__(self):
        self._lexicon = {}

    def add_term(self, term: str, document_frequency: int) -> None:
        """
        Adds or updates a term in the lexicon with its document frequency.
        If the term exists, adds to its current document frequency.

        Args:
            term (str): The term to add.
            document_frequency (int): The number of documents the term appears in.
        """
        self._lexicon[term] = self._lexicon.get(term, 0) + document_frequency

    def get_term_info(self, term: str) -> int:
        """
        Retrieves metadata for a given term.

        Args:
            term (str): The term to look up.

        Returns:
            dict: A dictionary with 'term_frequency' or None if the term is not found.
        """
        return self._lexicon.get(term)

    def get_all_terms(self) -> List[str]:
        """
        Returns all terms in the lexicon.

        Returns:
            List[str]: A list of all terms in the lexicon.
        """
        return list(self._lexicon.keys())

    def write_to_file(self, filename: str) -> None:
        """
        Writes the lexicon to a file for persistent storage.

        Args:
            filename (str): The path of the file to write the lexicon to.
        """
        with open(filename, "w") as f:
            for term, document_frequency in self._lexicon.items():
                f.write(f"{term} {document_frequency}\n")

    @staticmethod
    def load_from_file(filename: str) -> 'Lexicon':
        """
        Loads the lexicon from a file.

        Args:
            filename (str): The path of the file to read the lexicon from.

        Returns:
            Lexicon: The loaded Lexicon object.
        """
        lexicon = Lexicon()
        with open(filename, "r") as f:
            for line in f:
                term, document_frequency = line.strip().split()
                lexicon.add_term(term, int(document_frequency))
        return lexicon

    # Function to build the lexicon from the inverted index
    def build_lexicon(self, inverted_index) -> dict:
        """Builds a lexicon from the inverted index.

        Args:
            inverted_index (InvertedIndex): The inverted index object.

        Returns:
            dict: A dictionary representing the lexicon, mapping each term to its metadata.
        """

        for term in inverted_index.get_terms():
            postings = inverted_index.get_postings(term)
            # Assuming you want to store information like term frequency and postings offset
            document_frequency = len(postings)
            self._lexicon[term] = {
                "term_frequency": document_frequency
            }
        return self._lexicon