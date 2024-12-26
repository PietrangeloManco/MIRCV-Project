import unittest

from src.Utils.Preprocessing import Preprocessing


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize preprocessor once for all test methods."""
        cls.preprocessor = Preprocessing()

    def test_clean_text(self):
        """Test the clean_text function with various input scenarios."""
        test_cases = [
            # Standard URLs
            ("Visit https://example.com", "visit"),
            ("Check www.example.org", "check"),
            ("Go to http://site.com/path", "go to"),
            # HTML and scripts
            ("<html>Test</html>", "test"),
            ("<script>alert('Hi');</script>", None),
            ("<style>.hidden{display:none;}</style>", None),
            # Punctuation and special characters
            ("Hello, world!", "hello world"),
            ("Special chars #$%&*@!", "special chars"),
            # Empty or invalid input
            ("", None),
            (None, None),
        ]

        for input_text, expected_output in test_cases:
            with self.subTest(input=input_text):
                self.assertEqual(self.preprocessor.clean_text(input_text), expected_output)

    def test_tokenize(self):
        """Test the tokenize method with various input scenarios."""
        test_cases = [
            # Basic tokenization
            ("This is a test.", ["This", "is", "test"]),
            ("Tokenize this text!", ["Tokenize", "this", "text"]),
            # Punctuation inside words
            ("word1, word2!", ["word1", "word2"]),
            ("Hello-world, split-me.", ["Hello", "world", "split", "me"]),
            # Numbers
            ("Remove 123 and 4567.", ["Remove", "and"]),
            # Mixed inputs
            ("Normal sentence here.", ["Normal", "sentence", "here"]),
            # Edge cases
            ("", []),
            ("A single letter a.", ["single", "letter"]),
            ("Just single letters: a b c.", ["Just", "single", "letters"]),
        ]

        for input_text, expected_tokens in test_cases:
            with self.subTest(input=input_text):
                self.assertEqual(self.preprocessor.tokenize(input_text), expected_tokens)

    def test_single_text_preprocess(self):
        """Test the complete preprocessing pipeline with various scenarios."""
        test_cases = [
            # Full preprocessing pipeline
            ("Visit https://example.com today!", ["visit", "today"]),
            ("Check www.example.org carefully.", ["check", "care"]),
            ("Numbers 123 and punctuation!", ["number", "punctuat"]),
            ("Punctuation-in words, like this.", ["punctuat", "word", "like"]),
        ]

        for input_text, expected_tokens in test_cases:
            with self.subTest(input=input_text):
                self.assertEqual(self.preprocessor.single_text_preprocess(input_text), expected_tokens)

    def test_vectorized_preprocess(self):
        """Test vectorized preprocessing with multiple inputs."""
        test_cases = [
            (
                [
                    "Visit https://example.com today.",
                    "Normal text with punctuation!",
                    "Remove numbers like 1234."
                ],
                [
                    ["visit", "today"],
                    ["normal", "text", "punctuat"],
                    ["remov", "number", "like"]
                ]
            ),
            (
                [
                    "Complex example: subdomain.example.com/path.",
                    "Another test with 5678!",
                    "Empty input case"
                ],
                [
                    ["complex", "exampl"],
                    ["anoth", "test"],
                    ["empti", "input", "case"]
                ]
            )
        ]

        for input_texts, expected_tokens in test_cases:
            with self.subTest(input=input_texts):
                self.assertEqual(self.preprocessor.vectorized_preprocess(input_texts), expected_tokens)


if __name__ == "__main__":
    unittest.main()
