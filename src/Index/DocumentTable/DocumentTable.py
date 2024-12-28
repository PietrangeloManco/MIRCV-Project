from typing import Dict


class DocumentTable:
    def __init__(self):
        # Dict
        self._document_table = {}

    def add_document(self, doc_id: int, length: int) -> None:
        """
        Adds a document to the document table with its length.

        Args:
            doc_id (int): The unique identifier of the document.
            length (int): The number of terms in the document.
        """
        self._document_table[doc_id] = length

    def get_document_length(self, doc_id: int) -> int:
        """
        Retrieves the length of a document.

        Args:
            doc_id (int): The document ID.

        Returns:
            int: The number of terms in the document, or 0 if the document does not exist.
        """
        return self._document_table.get(doc_id, 0)

    def get_all_documents(self) -> Dict[int, int]:
        """
        Returns all documents with their lengths.

        Returns:
            Dict[int, int]: A dictionary where keys are document IDs and values are document lengths.
        """
        return self._document_table

    def write_to_file(self, filename: str) -> None:
        """
        Writes the document table to a file for persistent storage.

        Args:
            filename (str): The path of the file to write the document table to.
        """
        with open(filename, "w") as f:
            for doc_id, length in self._document_table.items():
                f.write(f"{doc_id} {length}\n")

    @staticmethod
    def load_from_file(filename: str) -> 'DocumentTable':
        """
        Loads the document table from a file.

        Args:
            filename (str): The path of the file to read the document table from.

        Returns:
            DocumentTable: The loaded DocumentTable object.
        """
        document_table = DocumentTable()
        with open(filename, "r") as f:
            for line in f:
                doc_id, length = map(int, line.strip().split())
                document_table.add_document(doc_id, length)
        return document_table