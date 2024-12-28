import time

from Index.DocumentTable.DocumentTable import DocumentTable
from Index.InvertedIndex.InvertedIndexBuilder import InvertedIndexBuilder
from Index.InvertedIndex.Merger import Merger
from Index.Lexicon.Lexicon import Lexicon
from Utils.CollectionLoader import CollectionLoader
from Utils.Preprocessing import Preprocessing


class IndexBuilderMain:
    def __init__(self):
        """
        Initializes all required components for index building.
        """
        # Initialize components
        self.collection_loader = CollectionLoader()
        self.preprocessing = Preprocessing()
        self.merger = Merger()
        self.document_table = DocumentTable()
        self.lexicon = Lexicon()

        # Initialize the InvertedIndexBuilder
        self.index_builder = InvertedIndexBuilder(
            collection_loader=self.collection_loader,
            preprocessing=self.preprocessing,
            merger=self.merger,
            document_table=self.document_table,
            lexicon=self.lexicon
        )

    def build_index(self) -> None:
        """
        Executes the index building process.
        """
        # Start the process
        print("Starting the Inverted Index Building process...")

        start_time = time.time()

        try:
            # Build the full index, lexicon, and document table
            print("Building the full index, lexicon, and document table...")
            self.index_builder.build_full_index()
            print("Index, lexicon, and document table built successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")
            return

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Index building completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    builder = IndexBuilderMain()
    builder.build_index()