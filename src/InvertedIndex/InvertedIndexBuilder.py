from src.InvertedIndex.InvertedIndex import InvertedIndex


class InvertedIndexBuilder:
    def __init__(self, collection_loader, preprocessing):
        self.collection_loader = collection_loader
        self.preprocessing = preprocessing
        self.inverted_index = InvertedIndex()

    def build_index(self):
        df = self.collection_loader.process_chunks()
        for doc_id, text in enumerate(df['text']):
            tokens = self.preprocessing.preprocess(text)
            for token in tokens:
                self.inverted_index.add_posting(token, doc_id)

    def get_index(self):
        return self.inverted_index
