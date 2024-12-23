import logging
import re
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from typing import List, Union, Optional

import pandas as pd
import unicodedata
from nltk import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm


class Preprocessing:
    def __init__(self, use_cache=True, stopwords_flag=True, stem_flag=True, min_word_length=2):
        # Initialize attributes
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.use_cache = use_cache
        self.stopwords_flag = stopwords_flag
        self.stem_flag = stem_flag
        self.min_word_length = min_word_length

        # Compile regex patterns
        self._noise_pattern = re.compile(
            r'\b(?:function|var|push|typeof|null|undefined'
            r'|ajax_fade|page_not_loaded|wpb-js-composer'
            r'|jquery|onclick|onload|script|style)\b'
        )

        # Enhanced URL pattern to catch more variants
        self._url_pattern = re.compile(
            r'''
            (?:https?://|www\.)?              # Optional protocol or www
            (?:[a-zA-Z0-9-]+\.)+             # Domain parts
            [a-zA-Z]{2,}                      # TLD
            (?:/[^\s<>]*)?                    # Optional path
            |                                 # OR
            (?:[a-zA-Z0-9-]+\.)+             # Domain without protocol
            (?:com|org|edu|gov|net|io|ai|app|dev|co|uk|us|eu|de|fr|it|es|nl)
            (?:/[^\s<>]*)?                    # Optional path
            ''', re.VERBOSE | re.IGNORECASE)

        self._html_pattern = re.compile(r'<[^>]+>')

    @staticmethod
    @lru_cache
    def clean_text(text: str) -> Optional[str]:
        if not isinstance(text, str) or not text.strip():
            return None

        # Normalize text
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove obvious script content
        text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL)

        # Remove specific noise patterns
        text = re.sub(
            r'\b(?:function|var|push|typeof|null|undefined'
            r'|ajax_fade|page_not_loaded|wpb-js-composer'
            r'|jquery|onclick|onload|script|style)\b',
            ' ', text
        )

        # Enhanced URL removal using the same pattern as in _url_pattern
        text = re.sub(
            r'''
            (?:https?://|www\.)?              # Optional protocol or www
            (?:[a-zA-Z0-9-]+\.)+             # Domain parts
            [a-zA-Z]{2,}                      # TLD
            (?:/[^\s<>]*)?                    # Optional path
            |                                 # OR
            (?:[a-zA-Z0-9-]+\.)+             # Domain without protocol
            (?:com|org|edu|gov|net|io|ai|app|dev|co|uk|us|eu|de|fr|it|es|nl)
            (?:/[^\s<>]*)?                    # Optional path
            ''',
            ' ', text, flags=re.VERBOSE | re.IGNORECASE)

        # Clean up remaining text
        text = re.sub(r'[^\w\s-]', ' ', text)  # Keep hyphens for compound words
        text = re.sub(r'\s+', ' ', text)

        cleaned = text.strip().lower()
        return cleaned if cleaned else None

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text and remove remnants of URLs or invalid tokens.
        """
        if not text:
            return []

        tokens = [
            token for token in text.split()
            if len(token) >= self.min_word_length
               and not token.isspace()
               and any(c.isalpha() for c in token)  # At least one letter
        ]

        # Additional URL fragment cleanup
        tokens = [token for token in tokens if not self._url_pattern.search(token)]
        return tokens

    # Rest of the class remains the same...
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.stop_words]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(word) for word in tokens]

    def _process_text_helper(self, args: tuple) -> List[str]:
        text, stopwords_flag, stem_flag = args
        return self.single_text_preprocess(text)

    def single_text_preprocess(self, text: str) -> List[str]:
        """
        Process a single text document with balanced filtering.
        """
        try:
            if not text:
                return []

            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return []

            tokens = self.tokenize(cleaned_text)

            if self.stopwords_flag:
                tokens = self.remove_stopwords(tokens)

            if self.stem_flag:
                tokens = self.stem_tokens(tokens)

            return tokens

        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            return []

    def vectorized_preprocess(self, texts: Union[pd.Series, List[str]]) -> List[List[str]]:
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        args = [(text, self.stopwords_flag, self.stem_flag) for text in texts]

        with Pool(cpu_count() - 1) as pool:
            all_preprocessed = list(tqdm(
                pool.imap(self._process_text_helper, args),
                total=len(texts),
            ))

        return all_preprocessed
