import re
from typing import List, Union
import pandas as pd
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from nltk import PorterStemmer
from nltk.corpus import stopwords
import logging
from tqdm import tqdm
import unicodedata

class Preprocessing:
    def __init__(self, use_cache: bool = True) -> None:
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.use_cache = use_cache

    @staticmethod
    @lru_cache(maxsize=1000)
    def clean_text(text: str) -> str:
        if not text:
            return ""

        # Normalize text to remove diacritical marks and ensure ASCII compatibility
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        # Remove non-alphanumeric characters and extra spaces
        text = re.sub(r'\W', ' ', text)  # Replace non-word characters with spaces
        text = re.sub(r'_', ' ', text)  # Replace underscores with spaces
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces into a single space

        return text.strip()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text:
            return []

        tokens = []
        current_entity = []

        for word in text.split():
            if word[0].isupper():
                current_entity.append(word)
            else:
                if current_entity:
                    tokens.append(' '.join(current_entity))
                    current_entity = []
                tokens.append(word)

        if current_entity:
            tokens.append(' '.join(current_entity))

        return tokens

    @staticmethod
    def lower_tokens(tokens: List[str]) -> List[str]:
        return [token.lower() for token in tokens]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.stop_words]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(word) for word in tokens]

    def _process_text_helper(self, args: tuple) -> List[str]:
        text, stopwords_flag, stem_flag = args
        return self._single_text_preprocess(text, stopwords_flag, stem_flag)

    def _single_text_preprocess(self, text: str, stopwords_flag: bool = True, stem_flag: bool = True) -> List[str]:
        try:
            if not text:
                return []

            text = self.clean_text(text)
            tokens = self.tokenize(text)
            tokens = self.lower_tokens(tokens)

            if stopwords_flag:
                tokens = self.remove_stopwords(tokens)

            if stem_flag:
                tokens = self.stem_tokens(tokens)

            return tokens

        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            return []

    def vectorized_preprocess(self,
                              texts: Union[pd.Series, List[str]],
                              stopwords_flag: bool = True,
                              stem_flag: bool = True) -> List[List[str]]:
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        args = [(text, stopwords_flag, stem_flag) for text in texts]

        with Pool(cpu_count() - 1) as pool:
            all_preprocessed = list(tqdm(
                pool.imap(self._process_text_helper, args),
                total=len(texts),
                desc="Preprocessing texts"
            ))

        return all_preprocessed
