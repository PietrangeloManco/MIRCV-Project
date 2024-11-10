from typing import Dict


# Document Table class
class DocumentTable:
    def __init__(self):
        self._document_table = {}

    def add_document(self, doc_id: int, length: int, offset: int) -> None:
        """
        Adds a document to the document table with its length and byte offset.

        Args:
            doc_id (int): The unique identifier of the document.
            length (int): The number of terms in the document (for TFIDF scoring).
            offset (int): The byte offset where the document starts in the original corpus.
        """
        self._document_table[doc_id] = {
            "length": length,
            "offset": offset
        }

    def get_document_length(self, doc_id: int) -> int:
        """
        Retrieves the length of a document.

        Args:
            doc_id (int): The document ID.

        Returns:
            int: The number of terms in the document.
        """
        return self._document_table.get(doc_id, {}).get("length", 0)

    def get_document_offset(self, doc_id: int) -> int:
        """
        Retrieves the byte offset of a document in the original corpus.

        Args:
            doc_id (int): The document ID.

        Returns:
            int: The byte offset where the document starts.
        """
        return self._document_table.get(doc_id, {}).get("offset", 0)

    def get_all_documents(self) -> Dict[int, Dict[str, int]]:
        """
        Returns all documents with their length and offset.

        Returns:
            Dict[int, Dict[str, int]]: A dictionary where keys are document IDs and values are dictionaries with length and offset.
        """
        return self._document_table

    def write_to_file(self, filename: str) -> None:
        """
        Writes the document table to a file for persistent storage.

        Args:
            filename (str): The path of the file to write the document table to.
        """
        with open(filename, "w") as f:
            for doc_id, data in self._document_table.items():
                length = data["length"]
                offset = data["offset"]
                f.write(f"{doc_id} {length} {offset}\n")

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
                doc_id, length, offset = map(int, line.strip().split())
                document_table.add_document(doc_id, length, offset)
        return document_table

    def build_document_table(self, documents_metadata) -> Dict[int, Dict[str, int]]:
        """
        Builds a document table using metadata about documents.

        Args:
            documents_metadata (list of dict): A list containing metadata for each document.

        Returns:
            Dict[int, Dict[str, int]]: A document table mapping document IDs to their metadata (length and offset).
        """
        for doc_metadata in documents_metadata:
            doc_id = doc_metadata["doc_id"]
            length = doc_metadata["length"]
            offset = doc_metadata["offset"]
            self.add_document(doc_id, length, offset)
        return self._document_table
