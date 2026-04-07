# MIRCV-Project

I built this project for a Multimedia Information Retrieval and Computer Vision course. In practice, it is a search engine designed for large document collections, and the whole point of the project was to implement the data structures, optimizations, and scoring techniques needed to make retrieval efficient at scale.

## What I Implemented

The project includes:

- inverted index construction,
- compressed postings,
- lexicon and document table management,
- query preprocessing and parsing,
- conjunctive and disjunctive retrieval,
- TF-IDF and BM25 scoring,
- evaluation over benchmark queries.

## Main Entry Points

- `src/IndexBuilderMain.py`: builds the core index structures.
- `src/SearchCLI.py`: interactive search CLI.
- `src/EvaluationMain.py`: evaluates the retrieval system.

## How To Run

1. Create a Python virtual environment.
2. Install the Python dependencies used by the project.
3. Download the NLTK stopwords corpus if needed.
4. Set `RESOURCES_PATH` to the `Files` directory of this repository, or edit `src/Utils/config.py`.
5. Run the desired entrypoint.

Example:

```bash
python src/IndexBuilderMain.py
python src/SearchCLI.py
python src/EvaluationMain.py
```

## Notes

- The project is built around the `Files/` resources directory, so the path configuration is important.
- The current code already exposes the search-engine components quite directly, which makes the repository useful both as a course project and as a compact IR systems implementation.
