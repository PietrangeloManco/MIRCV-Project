import unittest

from src.Utils.Preprocessing import Preprocessing


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize preprocessor once for all test methods."""
        cls.preprocessor = Preprocessing()

    def test_clean_text(self):
        """Test the clean_text function with various URL patterns."""
        test_cases = [
            # Standard URLs
            ("Visit https://example.com", "visit"),
            ("Check www.example.org", "check"),
            ("Go to http://site.com/path", "go to"),
        ]

        for input_text, expected_output in test_cases:
            with self.subTest(input=input_text):
                self.assertEqual(self.preprocessor.clean_text(input_text), expected_output)

    def test_tokenize(self):
        """Test the tokenize method with URL scenarios."""
        test_cases = [
            # Standard URLs
            ("visit https://example.com today", ["visit", "today"]),
            ("check www.example.org please", ["check", "please"]),
        ]

        for input_text, expected_tokens in test_cases:
            with self.subTest(input=input_text):
                self.assertEqual(self.preprocessor.tokenize(input_text), expected_tokens)

    def test_single_text_preprocess(self):
        """Test the complete preprocessing pipeline with URLs."""
        test_cases = [
            # Standard URLs
            ("Visit https://example.com today!", ["visit", "today"]),
            ("Check www.example.org carefully", ["check", "care"]),
        ]

        for input_text, expected_tokens in test_cases:
            with self.subTest(input=input_text):
                self.assertEqual(self.preprocessor.single_text_preprocess(input_text), expected_tokens)

    def test_vectorized_preprocess(self):
        """Test vectorized preprocessing with URLs."""
        test_cases = [
            (
                [
                    "Visit https://example.com today",
                    "Check example.org tomorrow",
                    "Normal text here"
                ],
                [
                    ["visit", "today"],
                    ["check", "tomorrow"],
                    ["normal", "text"]
                ]
            ),
            (
                [
                    "Multiple sites: example.com example.org",
                    "Complex URL: subdomain.example.com/path?param=1",
                    "No URLs in this one"
                ],
                [
                    ["multipl", "site"],
                    ["complex", "url"],
                    ["url", "one"]
                ]
            )
        ]

        for input_texts, expected_tokens in test_cases:
            with self.subTest(input=input_texts):
                self.assertEqual(self.preprocessor.vectorized_preprocess(input_texts), expected_tokens)


if __name__ == "__main__":
    unittest.main()

