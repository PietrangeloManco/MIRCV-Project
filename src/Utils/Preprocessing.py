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
    # Compile regex patterns as class variables to avoid repetition
    NOISE_PATTERN = re.compile(
        r'\b(?:function|var|push|typeof|null|undefined'
        r'|ajax_fade|page_not_loaded|wpb-js-composer'
        r'|jquery|onclick|onload|script|style)\b'
    )

    URL_PATTERN = re.compile(
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

    HTML_PATTERN = re.compile(r'<[^>]+>')
    SCRIPT_STYLE_PATTERN = re.compile(r'<(script|style)[^>]*>.*?</\1>', re.DOTALL)
    NON_WORD_PATTERN = re.compile(r'[^\w\s-]')
    WHITESPACE_PATTERN = re.compile(r'\s+')

    def __init__(self, use_cache: bool = True, stopwords_flag: bool = True,
                 stem_flag: bool = True, min_word_length: int = 2):
        """
        Preprocessing class initialization.

        Args:
            use_cache(bool): Flag to decide if using the cache or not. Default is true.
            stopwords_flag(bool): Flag to decide if performing stopwords removal or not. Default is true.
            stem_flag(bool): Flag to decide if performing stepping or not. Default is true.
            min_word_length(int): Minimum valid word length. Default is 2.
        """
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.use_cache = use_cache
        self.stopwords_flag = stopwords_flag
        self.stem_flag = stem_flag
        self.min_word_length = min_word_length

    @staticmethod
    @lru_cache
    def clean_text(text: str) -> Optional[str]:
        """
        Method to perform text cleaning.

        Args:
            text(str): The text to clean.

        Returns:
            Optional[str]: If something is left, it's returned.
        """
        if not isinstance(text, str) or not text.strip():
            return None

        # Normalize text
        text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')
                .decode('utf-8', 'ignore'))

        # Remove script and style tags with their content
        text = Preprocessing.SCRIPT_STYLE_PATTERN.sub(' ', text)

        # Remove all HTML tags and their content
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove noise and URLs
        text = Preprocessing.NOISE_PATTERN.sub(' ', text)
        text = Preprocessing.URL_PATTERN.sub(' ', text)

        # Clean up remaining text
        text = Preprocessing.NON_WORD_PATTERN.sub(' ', text)
        text = Preprocessing.WHITESPACE_PATTERN.sub(' ', text)

        # Lower and strip
        cleaned = text.strip().lower()
        return cleaned if cleaned else None

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text, split words connected by punctuation, and remove numbers.

        Args:
            text(str): Text to tokenize.

        Returns:
            List[str]: A list of cleaned tokens.
        """
        if not text:
            return []

        # Split text into tokens using non-word boundaries, preserving standalone words
        tokens = re.split(r'\W+', text)

        # Filter tokens based on rules:
        # - Remove tokens shorter than the minimum word length
        # - Exclude pure numbers
        # - Ensure tokens contain at least one alphabetic character
        tokens = [
            token for token in tokens
            if len(token) >= self.min_word_length
               and any(c.isalpha() for c in token)  # At least one letter
               and not token.isdigit()  # Exclude pure numbers
        ]

        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Stopwords removal method.

        Args:
            tokens(List[str]): List of tokens to remove stopwords from.

        Returns:
            List[str]: List of tokens without stopwords.
        """
        return [word for word in tokens if word not in self.stop_words]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Stemming method.

        Args:
            tokens(List[str]): List of tokens to stem.

        Returns:
            List[str]: List of stemmed tokens.
        """
        return [self.stemmer.stem(word) for word in tokens]

    def _process_text_helper(self, args: tuple) -> List[str]:
        """
        Helper function to perform the parallel vectorized preprocessing.

        Args:
            args(tuple): A tuple containing the arguments to pass to the single text preprocess method.

        Returns: A list of preprocessed tokens.
        """
        text, stopwords_flag, stem_flag = args
        return self.single_text_preprocess(text)

    def single_text_preprocess(self, text: str) -> List[str]:
        """
        Process a single text document.

        Args:
            text(str): Text to process.

        Returns:
            List[str]: List of preprocessed tokens.
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
        """
        Method to perform an efficient vectorized preprocessing.

        Args:
            texts(List[str]): A list of texts to preprocess.

        Returns:
            List[List[str]]: A list of lists of tokens, one for each input text.
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        args = [(text, self.stopwords_flag, self.stem_flag) for text in texts]

        with Pool(cpu_count() - 1) as pool:
            all_preprocessed = list(tqdm(
                pool.imap(self._process_text_helper, args),
                total=len(texts),
            ))

        return all_preprocessed
