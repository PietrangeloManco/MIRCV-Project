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
            ("new york city", ["new york city"]),
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

    def test_vectorized_preprocessing(self):
        """Test vectorized preprocessing with multiple texts"""
        sample_texts = [
            "First document with some text",
            "Second document with different words",
            "Third document to test preprocessing"
        ]

        # Test with different preprocessing configurations
        preprocessing_configs = [
            {"stopwords_flag": True, "stem_flag": True},
            {"stopwords_flag": False, "stem_flag": False}
        ]

        for config in preprocessing_configs:
            with self.subTest(**config):
                processed_texts = self.preprocessor.vectorized_preprocess(
                    sample_texts,
                    stopwords_flag=config["stopwords_flag"],
                    stem_flag=config["stem_flag"]
                )

                # Verify output is a list of lists
                self.assertIsInstance(processed_texts, list)
                self.assertEqual(len(processed_texts), len(sample_texts))

                # Verify each processed text is a list of strings
                for processed_text in processed_texts:
                    self.assertIsInstance(processed_text, list)
                    for token in processed_text:
                        self.assertIsInstance(token, str)

if __name__ == "__main__":
    unittest.main()