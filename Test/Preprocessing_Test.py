import unittest
from Utils.Preprocessing import Preprocessing

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = Preprocessing()

    def test_clean_text(self):
        self.assertEqual(self.preprocessor.clean_text("Hello, World!"), "hello  world ")
        self.assertEqual(self.preprocessor.clean_text(""), "")
        self.assertEqual(self.preprocessor.clean_text("_"), " ")

    def test_tokenize(self):
        self.assertEqual(self.preprocessor.tokenize("hello world"), ["hello", "world"])
        self.assertEqual(self.preprocessor.tokenize("new york"), ["new york"])

    def test_remove_stopwords(self):
        tokens = ["this", "is", "a", "test"]
        self.assertEqual(self.preprocessor.remove_stopwords(tokens), ["test"])

    def test_stem_tokens(self):
        tokens = ["running", "jumps"]
        self.assertEqual(self.preprocessor.stem_tokens(tokens), ["run", "jump"])

    def test_preprocess(self):
        sample_text = "This is an example sentence to demonstrate text preprocessing."
        processed_text = self.preprocessor.preprocess(sample_text)
        expected_output = ["exampl", "sentenc", "demonstr", "text", "preprocess"]
        self.assertEqual(processed_text, expected_output)

if __name__ == "__main__":
    unittest.main()


