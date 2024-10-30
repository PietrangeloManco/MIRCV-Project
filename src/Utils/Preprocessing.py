import re
from typing import List
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Preprocessing class
class Preprocessing:
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.nlp = spacy.load("en_core_web_sm")

    # Method to remove non-alphanumeric characters
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'_', ' ', text)
        text = text.lower()
        return text

    # Method to tokenize the text, using a NER procedure
    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
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

    # Method to remove stopwords
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return filtered_tokens

    # Method to stem the tokens
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        return stemmed_tokens

    # Full pipeline method
    def preprocess(self, text: str) -> List[str]:
        try:
            text = self.clean_text(text)
            tokens = self.tokenize(text)
            tokens = self.remove_stopwords(tokens)
            tokens = self.stem_tokens(tokens)
            return tokens
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return []