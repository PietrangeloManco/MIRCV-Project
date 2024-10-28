import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class Preprocessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def clean_text(text):
        if not text:
            return ""
        # Remove non-alphanumeric characters
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'_', ' ', text)
        # Convert to lowercase
        text = text.lower()
        return text

    def tokenize(self, text):
        if not text:
            return []
        # Tokenize the text
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if token.ent_type_ in ["GPE", "ORG", "PERSON"]:
                if token.ent_iob_ == 'B':  # Beginning of an entity
                    tokens.append(token.text)
                else:
                    tokens[-1] += ' ' + token.text
            else:
                tokens.append(token.text)
        return tokens

    def remove_stopwords(self, tokens):
        if not tokens:
            return []
        # Remove stop words
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return filtered_tokens

    def stem_tokens(self, tokens):
        if not tokens:
            return []
        # Stem the tokens
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        return stemmed_tokens

    def preprocess(self, text):
        try:
            # Full preprocessing pipeline
            text = self.clean_text(text)
            tokens = self.tokenize(text)
            tokens = self.remove_stopwords(tokens)
            tokens = self.stem_tokens(tokens)
            return tokens
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return []