The program includes several tests that can be ran to evaluate single modules' functionalities.
There are 3 main runnable modules:
-EvaluationMain.py performs an evaluation of the Retrieval System computing the NDCG over benchmark queries and assessements.
-IndexBuilderMain.py performs the construction from scratch of the InvertedIndex, Lexicon and DocumentTable structures.
-SearchCLI.py is a command line interface that allows the user to input queries, choose between conjunctive and disjunctive modes, 
  select the desired scoring function between TFIDF and BM25, and get a list of 10 document ids, ranked in descending order of relevance.
  After the list is returned, the program asks for a new query, until 'exit' is inserted.
