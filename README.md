# Program Overview
The config.py file requires the user to insert the path to resources in his system.
The program includes several tests that can be run to evaluate single modules' functionalities. There are 3 main runnable modules:

- **`EvaluationMain.py`**: Performs an evaluation of the Retrieval System computing the NDCG over benchmark queries and assessments.
- **`IndexBuilderMain.py`**: Performs the construction from scratch of the InvertedIndex, Lexicon, and DocumentTable structures.
- **`SearchCLI.py`**: A command line interface that allows the user to:
  - Input queries.
  - Choose between conjunctive and disjunctive modes.
  - Select the desired scoring function between TFIDF and BM25.
  - Get a list of 10 document IDs, ranked in descending order of relevance.

  After the list is returned, the program asks for a new query, until `exit` is inserted.

