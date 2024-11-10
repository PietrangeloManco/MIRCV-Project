import unittest
from src.Utils.Preprocessing import Preprocessing

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        """Initialize preprocessor before each test method"""
        self.preprocessor = Preprocessing()

    def test_clean_text_basic(self):
        """Test basic text cleaning functionality"""
        test_cases = [
            ("Hello, World!", "hello world"),
            ("", ""),
            ("_", ""),
            ("Multiple   Spaces", "multiple spaces"),
            ("Special@#$Chars", "special chars")
        ]

        for input_text, expected_output in test_cases:
            with self.subTest(input=input_text):
                self.assertEqual(self.preprocessor.clean_text(input_text), expected_output)

    def test_tokenize_scenarios(self):
        """Test various tokenization scenarios"""
        test_cases = [
            ("hello world", ["hello", "world"]),
            ("New York City", ["New", "York", "City"]),
            ("", []),
            ("OpenAI is an AI research company",
             ["OpenAI", "is", "an", "AI", "research", "company"])
        ]

        for input_text, expected_tokens in test_cases:
            with self.subTest(input=input_text):
                self.assertEqual(self.preprocessor.tokenize(input_text), expected_tokens)

    def test_stopwords_removal(self):
        """Comprehensive stopwords removal test"""
        test_cases = [
            (["this", "is", "a", "test"], ["test"]),
            (["the", "quick", "brown", "fox"], ["quick", "brown", "fox"]),
            ([], []),
            (["is", "are", "was"], [])
        ]

        for input_tokens, expected_tokens in test_cases:
            with self.subTest(input=input_tokens):
                self.assertEqual(self.preprocessor.remove_stopwords(input_tokens), expected_tokens)

    def test_stemming(self):
        """Comprehensive token stemming test"""
        test_cases = [
            (["running", "jumps"], ["run", "jump"]),
            (["quickly", "quietly"], ["quickli", "quietli"]),
            ([], []),
            (["cats", "dogs", "mice"], ["cat", "dog", "mice"])
        ]

        for input_tokens, expected_tokens in test_cases:
            with self.subTest(input=input_tokens):
                self.assertEqual(self.preprocessor.stem_tokens(input_tokens), expected_tokens)

if __name__ == "__main__":
    unittest.main()