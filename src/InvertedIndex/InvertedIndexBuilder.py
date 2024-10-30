from src.InvertedIndex.InvertedIndex import InvertedIndex
from src.Utils.CollectionLoader import CollectionLoader
from src.Utils.Preprocessing import Preprocessing

# Class to actually build the Inverted Index out of the input collection
class InvertedIndexBuilder:
    def __init__(self, collection_loader: CollectionLoader, preprocessing: Preprocessing) -> None:
        self.collection_loader = collection_loader
        self.preprocessing = preprocessing
        self.inverted_index = InvertedIndex()

    # Method to build the full index
    def build_full_index(self) -> None:
        df = self.collection_loader.process_chunks()
        for index, row in df.iterrows():
            doc_id = row[0]
            text = row[1]
            tokens = self.preprocessing.preprocess(text)
            for token in tokens:
                self.inverted_index.add_posting(token, doc_id)

    # Method to build a partial index for testing purposes. Default is 10 lines.
    def build_partial_index(self, num_lines: int = 10) -> None:
        df = self.collection_loader.sample_lines(num_lines)
        print(f"Processed chunks: {df}")
        for index, row in df.iterrows():
            doc_id = row[0]
            text = row[1]
            tokens = self.preprocessing.preprocess(text)
            print(f"Document ID: {doc_id}, Tokens: {tokens}")
            for token in tokens:
                self.inverted_index.add_posting(token, doc_id)
                print(f"Added token: {token} for document ID: {doc_id}")

    # Method to get the index
    def get_index(self) -> InvertedIndex:
        return self.inverted_index
