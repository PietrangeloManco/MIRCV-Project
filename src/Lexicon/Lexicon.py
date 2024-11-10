# Lexicon structure class
from typing import List


class Lexicon:
    def __init__(self):
        self._lexicon = {}

    def add_term(self, term: str, position: int, term_frequency: int) -> None:
        """
        Adds a term to the lexicon with its position in the index file and the term frequency.

        Args:
            term (str): The term to add.
            position (int): The byte position in the index file where the term's postings start.
            term_frequency (int): The total number of occurrences of the term across all documents.
        """
        self._lexicon[term] = {"position": position, "term_frequency": term_frequency}

    def get_term_info(self, term: str) -> dict:
        """
        Retrieves metadata for a given term.

        Args:
            term (str): The term to look up.

        Returns:
            dict: A dictionary with 'position' and 'term_frequency' or None if the term is not found.
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
            for term, metadata in self._lexicon.items():
                position = metadata["position"]
                term_frequency = metadata["term_frequency"]
                f.write(f"{term} {position} {term_frequency}\n")

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
                term, position, term_frequency = line.strip().split()
                lexicon.add_term(term, int(position), int(term_frequency))
        return lexicon

    # Function to build the lexicon from the inverted index
    def build_lexicon(self, inverted_index) -> dict:
        """Builds a lexicon from the inverted index.

        Args:
            inverted_index (InvertedIndex): The inverted index object.

        Returns:
            dict: A dictionary representing the lexicon, mapping each term to its metadata.
        """
        offset = 0  # This could be the byte offset if storing the index in a file

        for term in inverted_index.get_terms():
            postings = inverted_index.get_postings(term)
            # Assuming you want to store information like term frequency and postings offset
            term_frequency = len(postings)
            self._lexicon[term] = {
                "offset": offset,
                "term_frequency": term_frequency
            }
            # Update the offset based on the number of postings (or the length of data if compressed)
            offset += len(postings)  # Adjust this as needed for your storage format

        return self._lexicon