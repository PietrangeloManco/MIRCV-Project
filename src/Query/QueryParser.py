from typing import List

from Utils.Preprocessing import Preprocessing


class QueryParser:
    def __init__(self, preprocessing: Preprocessing):
        """
        Initializes the QueryParser.

        Args:
            preprocessing: An instance of the Preprocessing class for text preprocessing.
        """
        self.preprocessing = preprocessing

    def parse(self, query: str) -> List[str]:
        """
        Parses and tokenizes the input query string.

        Args:
            query(str): The query string.

        Returns:
             A list of query terms after tokenization and preprocessing.
        """
        if not query:
            return []

        # Preprocess the query: clean, tokenize, remove stopwords, and stem
        tokens = self.preprocessing.single_text_preprocess(query)

        return tokens
