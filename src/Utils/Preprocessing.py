import re
from typing import List, Union
import spacy
import pandas as pd
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from nltk import PorterStemmer
from nltk.corpus import stopwords
import logging

class Preprocessing:
    def __init__(self, use_cache: bool = True) -> None:
        # Lazy load NLP model to improve startup time
        self._nlp = None
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.use_cache = use_cache

    @property
    def nlp(self):
        # Lazy loading of spaCy model
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    @staticmethod
    @lru_cache(maxsize=1000)
    def clean_text(text: str) -> str:
        if not text:
            return ""
        # Compile regex patterns once
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'_', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        # Use spaCy's built-in NER more efficiently
        doc = self.nlp(text)
        tokens = []
        current_entity = []

        for token in doc:
            if token.ent_type_ in ["GPE", "ORG", "PERSON"]:
                current_entity.append(token.text)
            else:
                # Add any accumulated entity first
                if current_entity:
                    tokens.append(' '.join(current_entity))
                    current_entity = []

                # Add current token
                tokens.append(token.text)

        # Add last entity if exists
        if current_entity:
            tokens.append(' '.join(current_entity))

        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.stop_words]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(word) for word in tokens]

    # Vectorized preprocessing method
    def vectorized_preprocess(self,
                              texts: Union[pd.Series, List[str]],
                              stopwords_flag: bool = True,
                              stem_flag: bool = True) -> List[List[str]]:
        # Parallel processing for large datasets
        num_cores = cpu_count() - 1

        # Convert to list if it's a pandas Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        # Use multiprocessing for large datasets
        with Pool(num_cores) as pool:
            # Parallel preprocessing
            preprocessed = pool.starmap(
                self._single_text_preprocess,
                [(text, stopwords_flag, stem_flag) for text in texts]
            )

        return preprocessed

    def _single_text_preprocess(self,
                                text: str,
                                stopwords_flag: bool = True,
                                stem_flag: bool = True) -> List[str]:
        try:
            # Reuse existing preprocessing steps
            text = self.clean_text(text)
            tokens = self.tokenize(text)

            if stopwords_flag:
                tokens = self.remove_stopwords(tokens)

            # Process tokens with spaCy to get POS tags and lemmas
            doc = self.nlp(" ".join(tokens))
            tokens = [token.lemma_ for token in doc]

            if stem_flag:
                tokens = self.stem_tokens(tokens)

            return tokens
        except Exception as e:
            # Log the error instead of printing
            logging.error(f"Error during preprocessing: {e}")
            return []
