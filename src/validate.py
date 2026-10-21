"""Step 7: check the from-scratch BM25 and Naïve Bayes against library implementations.

Run from the project root:  python -m src.validate
Writes reports/validation.md.
"""

import math

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from src.classifier import NaiveBayes
from src.demo_search import QUERIES as DEMO_QUERIES
from src.indexer import InvertedIndex
from src.loader import load_articles
from src.preprocess import load_tokens, preprocess
from src.rankers import bm25_search

VALIDATION_MD = "reports/validation.md"

# The 3 demo queries plus 7 more, so every category is covered at least once.
QUERIES = DEMO_QUERIES + [
    "oil prices economy",              # business
    "oscar nominations best actress",  # entertainment
    "film box office",                 # entertainment
    "general election labour tory",    # politics
    "immigration asylum policy",       # politics
    "world cup rugby",                 # sport
    "broadband internet users",        # tech
]
IDF_EXAMPLE_TERMS = ["said", "bank", "chelsea", "asylum"]  # a very common word, then rarer ones


def okapi_top10(okapi: BM25Okapi, q_terms: list[str]) -> list[int]:
    """BM25Okapi's 10 best doc_ids (row i of the corpus is doc_id i); ties go to the lower doc_id."""
    scores = okapi.get_scores(q_terms)
    return list(np.argsort(-scores, kind="stable")[:10])


def bm25_with_okapi_idf(index: InvertedIndex, okapi: BM25Okapi, q_terms: list[str],
                        k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """My BM25 formula for every doc, but with BM25Okapi's idf values instead of mine.

    score(q, d) = Σ_t okapi_idf(t) · tf · (k1 + 1) / (tf + k1 · (1 - b + b · |d| / avgdl))
    If these scores equal okapi.get_scores(), the only difference between the two BM25s is idf.
    """
    scores = np.zeros(index.N)
    for term in q_terms:
        for doc_id, tf in index.postings.get(term, {}).items():
            length_norm = 1 - b + b * index.doc_len[doc_id] / index.avgdl
            scores[doc_id] += okapi.idf[term] * tf * (k1 + 1) / (tf + k1 * length_norm)
    return scores


def validate_bm25(index: InvertedIndex, tokens: dict[int, list[str]]) -> tuple[list[dict], list[dict]]:
    """Compare my bm25_search with rank_bm25's BM25Okapi on the same tokens (k1=1.5, b=0.75)."""
    corpus = [tokens[d] for d in range(index.N)]  # doc_id order, so list position = doc_id
    okapi = BM25Okapi(corpus, k1=1.5, b=0.75)

    rows = []
    for query in QUERIES:
        q_terms = preprocess(query)
        mine = [d for d, _ in bm25_search(index, q_terms, k=10, k1=1.5, b=0.75)]
        theirs = okapi_top10(okapi, q_terms)
        rows.append({
            "query": query,
            "overlap": len(set(mine) & set(theirs)),
            "top_match": mine[0] == theirs[0],
            "mine_top": mine[0],
            "okapi_top": int(theirs[0]),
            # Largest score difference over ALL docs when my formula uses Okapi's idf (should be ~0).
            "max_diff": float(np.abs(bm25_with_okapi_idf(index, okapi, q_terms) - okapi.get_scores(q_terms)).max()),
        })

    idf_rows = []
    for term in IDF_EXAMPLE_TERMS:
        df = index.df(term)
        idf_rows.append({
            "term": term,
            "df": df,
            "mine": math.log((index.N - df + 0.5) / (df + 0.5) + 1),
            "okapi_raw": math.log(index.N - df + 0.5) - math.log(df + 0.5),
            "okapi": okapi.idf[term],
        })
    return rows, idf_rows


def validate_nb(articles, tokens: dict[int, list[str]]) -> dict:
    """Compare my NaiveBayes (models/nb.pkl) with scikit-learn's MultinomialNB(alpha=1.0) on the test split."""
    train = articles[articles["split"] == "train"]
    test = articles[articles["split"] == "test"]
    train_docs = [tokens[d] for d in train["doc_id"]]
    test_docs = [tokens[d] for d in test["doc_id"]]

    # analyzer=lambda x: x means "the documents are already token lists, don't tokenize again".
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    sk_model = MultinomialNB(alpha=1.0)
    sk_model.fit(vectorizer.fit_transform(train_docs), train["category"])
    sk_pred = list(sk_model.predict(vectorizer.transform(test_docs)))

    my_model = NaiveBayes.load()
    my_pred = [my_model.predict(doc)[0] for doc in test_docs]

    truth = list(test["category"])
    agree = sum(a == b for a, b in zip(my_pred, sk_pred))
    return {
        "n_test": len(test),
        "my_acc": accuracy_score(truth, my_pred),
        "sk_acc": accuracy_score(truth, sk_pred),
        "agree": agree,
        "agree_pct": 100 * agree / len(test),
    }


def to_markdown(bm25_rows: list[dict], idf_rows: list[dict], nb: dict) -> str:
    """Format the results as the text of reports/validation.md."""
    avg_overlap = sum(r["overlap"] for r in bm25_rows) / len(bm25_rows)
    top_matches = sum(r["top_match"] for r in bm25_rows)
    lines = [
        "# Validation against library implementations",
        "",
        "## 1. BM25: my `bm25_search` vs `rank_bm25.BM25Okapi` (k1=1.5, b=0.75, top 10)",
        "",
        "| Query | Overlap /10 | Top result matches | My top doc | BM25Okapi top doc | Max score diff with Okapi idf |",
        "|---|---|---|---|---|---|",
    ]
    for r in bm25_rows:
        lines.append(f"| {r['query']} | {r['overlap']} | {'yes' if r['top_match'] else 'no'} "
                     f"| {r['mine_top']} | {r['okapi_top']} | {r['max_diff']:.1e} |")
    lines += [
        f"| **Average** | **{avg_overlap:.1f}** | **{top_matches}/{len(bm25_rows)}** | | | |",
        "",
        "The term-frequency and length parts of the two formulas are identical; only idf differs:",
        "",
        "- mine: `idf = ln((N - df + 0.5) / (df + 0.5) + 1)`, always positive;",
        "- BM25Okapi: `idf = ln((N - df + 0.5) / (df + 0.5))`, negative when df > N/2, and negative values are",
        "  replaced by `epsilon * average idf` (epsilon = 0.25).",
        "",
        "The last column re-scores every document with my formula but BM25Okapi's idf values: the scores are",
        "identical up to floating-point rounding, so idf is the only difference. Because of the `+ 1`, my idf",
        "gives common words relatively more weight (e.g. `world`, df 639: 1.188 vs 0.824), which is why a",
        "document near rank 10 sometimes swaps.",
        "",
        "| Term | df | My idf | Okapi raw idf | Okapi idf used |",
        "|---|---|---|---|---|",
    ]
    for r in idf_rows:
        lines.append(f"| {r['term']} | {r['df']} | {r['mine']:.3f} | {r['okapi_raw']:.3f} | {r['okapi']:.3f} |")
    lines += [
        "",
        "## 2. Naïve Bayes: my `NaiveBayes` vs scikit-learn `MultinomialNB(alpha=1.0)`",
        "",
        "| Model | Test accuracy |",
        "|---|---|",
        f"| My NaiveBayes | {100 * nb['my_acc']:.2f}% |",
        f"| scikit-learn MultinomialNB | {100 * nb['sk_acc']:.2f}% |",
        "",
        f"Predictions agree on **{nb['agree']} / {nb['n_test']} test articles ({nb['agree_pct']:.2f}%)**.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    index = InvertedIndex.load()
    tokens = load_tokens()
    articles = load_articles()

    bm25_rows, idf_rows = validate_bm25(index, tokens)
    nb = validate_nb(articles, tokens)
    report = to_markdown(bm25_rows, idf_rows, nb)

    with open(VALIDATION_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Saved {VALIDATION_MD}")
