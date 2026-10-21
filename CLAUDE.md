# Intelligent News Retrieval and Classification System

College IR project: search BBC News articles (data/bbc/, 2,225 articles in 5 categories) by topic using an
inverted index with TF-IDF, BM25 and a query-likelihood unigram language model, and classify articles by
category with Naïve Bayes. The student must be able to explain every line in a viva.

## Project rules

- Write the inverted index, TF-IDF, BM25, the unigram language model, Naïve Bayes and the evaluation metrics
  from scratch in plain Python (math, collections and numpy are fine). Use scikit-learn only for the
  train/test split, classification metrics and validation, and rank_bm25 only for validation.
- Every text, document or query, goes through preprocess() in src/preprocess.py. Never write a second tokenizer.
- Compute probabilities in log space.
- Defaults: BM25 k1=1.5, b=0.75; LM Dirichlet mu=2000; Jelinek-Mercer lambda=0.1 on the collection model;
  Naïve Bayes add-one smoothing (alpha=1.0); stratified 80/20 split with random_state=42.
- Shared files: data/articles.csv, data/tokens.pkl, models/index.pkl, models/nb.pkl, data/queries.csv,
  data/pool_to_judge.csv, data/qrels.csv, reports/validation.md, reports/results.md, reports/viva_notes.md.
  Later steps load these files instead of recomputing them.
- Run modules as `python -m src.<name>` from the project root with the .venv interpreter
  (Windows: `.venv\Scripts\python.exe -m src.<name>`). Tests live in tests/ and run with pytest
  (`.venv\Scripts\python.exe -m pytest`).
- Keep code simple: short functions, type hints, a docstring with the formula on every scoring function, and
  comments a student can follow. No extra frameworks or classes beyond what a step asks for.
- At the end of every step: run its tests, update that step's section in PROGRESS.md, append a plain-English
  section to reports/viva_notes.md, and commit with the message "Step N: <name>".
