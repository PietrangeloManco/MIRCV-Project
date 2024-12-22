import unittest
import os

from DocumentTable.DocumentTable import DocumentTable


class TestDocumentTable(unittest.TestCase):

    def setUp(self):
        """Sets up a new DocumentTable instance for each test."""
        self.document_table = DocumentTable()

    def test_add_document(self):
        """Test adding a document to the document table."""
        self.document_table.add_document(1, 100)
        doc_data = self.document_table.get_all_documents()

        # Check if document with doc_id 1 exists
        self.assertIn(1, doc_data)
        self.assertEqual(doc_data[1], 100)

    def test_get_document_length(self):
        """Test retrieving the length of a document."""
        self.document_table.add_document(2, 200)
        length = self.document_table.get_document_length(2)
        self.assertEqual(length, 200)

    def test_get_non_existing_document(self):
        """Test retrieving a non-existing document."""
        # The document with id 4 doesn't exist
        length = self.document_table.get_document_length(4)

        # Should return default values: 0 for length and offset
        self.assertEqual(length, 0)

    def test_write_and_load_from_file(self):
        """Test writing the document table to a file and loading it back."""
        self.document_table.add_document(5, 250)
        self.document_table.add_document(6, 300)

        # Write to file
        filename = "document_table_test.txt"
        self.document_table.write_to_file(filename)

        # Load the document table from the file
        loaded_document_table = DocumentTable.load_from_file(filename)

        # Check if the data is correctly loaded
        loaded_data = loaded_document_table.get_all_documents()

        self.assertIn(5, loaded_data)
        self.assertIn(6, loaded_data)
        self.assertEqual(loaded_data[5], 250)
        self.assertEqual(loaded_data[6], 300)

        # Clean up the test file
        os.remove(filename)

    def test_build_document_table(self):
        """Test building a document table from metadata."""
        documents_metadata = [
            {"doc_id": 7, "length": 400},
            {"doc_id": 8, "length": 500}
        ]

        self.document_table.build_document_table(documents_metadata)
        doc_data = self.document_table.get_all_documents()

        self.assertIn(7, doc_data)
        self.assertIn(8, doc_data)
        self.assertEqual(doc_data[7], 400)
        self.assertEqual(doc_data[8], 500)

    def tearDown(self):
        """Clean up after each test."""
        del self.document_table


# Run the tests
if __name__ == '__main__':
    unittest.main()
