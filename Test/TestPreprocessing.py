import unittest
from src.Utils.Preprocessing import Preprocessing


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        """Initialize preprocessor before each test method."""
        self.preprocessor = Preprocessing()

    def test_clean_text(self):
        """Test the clean_text function with various URL patterns."""
        test_cases = [
            # Standard URLs
            ("Visit https://example.com", "visit"),
            ("Check www.example.org", "check"),
            ("Go to http://site.com/path", "go to"),

            # URLs without protocol
            ("Visit example.com", "visit"),
            ("Check site.co.uk", "check"),
            ("Go to mysite.org/path", "go to"),

            # Complex URLs
            ("Visit subdomain.example.com/path?param=1", "visit"),
            ("Check my-site.io/api/v1", "check"),
            ("Go to app.dev/dashboard", "go to"),

            # Multiple URLs
            ("Check example.com and example.org", "check and"),
            ("Visit site.io or app.dev", "visit or"),

            # URLs with text
            ("Website: my-company.com/about", "website"),
            ("Contact: contact.example.org", "contact"),

            # Common TLDs
            ("site.com site.net site.org", None),
            ("domain.io domain.ai domain.app", None),
            ("page.co.uk page.us page.eu", None),

            # Edge cases
            ("", None),  # Empty string
            ("No URLs here!", "no urls here"),
            ("email@domain.com", "email"),  # Email addresses
            ("Multiple   spaces   here", "multiple spaces here"),

            # URLs with special characters
            ("site.com/path#section", None),
            ("domain.org/path?param=value", None),

            # Mixed content
            ("Text before example.com text after", "text before text after"),
            ("<html>domain.com</html>", None),
            ("URL: my-site.io, continues", "url continues")
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

            # URLs without protocol
            ("visit example.com today", ["visit", "today"]),
            ("check site.co.uk please", ["check", "please"]),

            # Mixed content
            ("before mysite.io after", ["before", "after"]),
            ("text app.dev more", ["text", "more"]),

            # Multiple URLs
            ("example.com example.org", []),
            ("site.io app.dev domain.com", []),

            # Edge cases
            ("", []),
            ("normal text here", ["normal", "text", "here"]),
            ("my-hyphenated-word", ["my-hyphenated-word"])
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

            # URLs without protocol
            ("Visit example.com tomorrow", ["visit", "tomorrow"]),
            ("Check site.co.uk please", ["check", "pleas"]),

            # Multiple URLs and mixed content
            ("First example.com then example.org", ["first"]),
            ("Between mysite.io and app.dev text", ["text"]),

            # Complex cases
            ("Running quickly to subdomain.example.com/path", ["run", "quickli"]),
            ("The site my-company.io/about is great", ["site", "great"]),

            # Normal text for comparison
            ("The quick brown fox", ["quick", "brown", "fox"]),
            ("Running and jumping", ["run", "jump"])
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