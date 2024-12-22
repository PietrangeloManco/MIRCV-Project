import time

from DocumentTable.DocumentTable import DocumentTable
from InvertedIndex.InvertedIndexBuilder import InvertedIndexBuilder
from InvertedIndex.Merger import Merger
from Lexicon.Lexicon import Lexicon
from Utils.CollectionLoader import CollectionLoader
from Utils.Preprocessing import Preprocessing


def main():

    # Initialize components
    collection_loader = CollectionLoader()
    preprocessing = Preprocessing()
    merger = Merger()
    document_table = DocumentTable()
    lexicon = Lexicon()

    # Initialize the InvertedIndexBuilder
    index_builder = InvertedIndexBuilder(
        collection_loader=collection_loader,
        preprocessing=preprocessing,
        merger=merger,
        document_table=document_table,
        lexicon=lexicon
    )

    # Start the process
    print("Starting the Inverted Index Building process...")

    start_time = time.time()

    try:
        # Build the full index, lexicon, and document table
        print("Building the full index, lexicon, and document table...")
        index_builder.build_full_index()

        print("Index, lexicon, and document table built successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        return

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Index building completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
