"""Step 8: build the pool of documents to judge for relevance (pooling the top 10 of all three rankers).

Run from the project root:  python -m src.pool
Writes data/pool_to_judge.csv; fill in its rel column by hand (0, 1 or 2).
"""

import pandas as pd

from src.indexer import InvertedIndex
from src.loader import load_articles
from src.preprocess import load_tokens, preprocess
from src.rankers import RANKERS, search

QUERIES_CSV = "data/queries.csv"
POOL_CSV = "data/pool_to_judge.csv"
POOL_DEPTH = 10  # take the top 10 of each ranker


def load_queries() -> pd.DataFrame:
    """Read data/queries.csv: query_id, query, intended_category."""
    return pd.read_csv(QUERIES_CSV, encoding="utf-8")


def pool_for_query(index: InvertedIndex, query: str, k: int = POOL_DEPTH) -> list[int]:
    """Union of the top-k doc_ids from every ranker, sorted by doc_id.

    Sorting by doc_id hides which ranker found each document, so the judge is not biased by rank.
    """
    pool: set[int] = set()
    for model in RANKERS:
        pool |= {doc_id for doc_id, _ in search(index, query, model, k)}
    return sorted(pool)


def rarest_term(index: InvertedIndex, q_terms: list[str]) -> str | None:
    """The query term with the highest idf, i.e. the lowest df > 0 (None if no term is in the index)."""
    known = [t for t in q_terms if index.df(t) > 0]
    if not known:
        return None
    return min(known, key=index.df)


def suggest_rel(doc_tokens: set[str], category: str, intended_category: str, rare_term: str | None) -> int:
    """Hint only: 1 if the doc is in the intended category AND contains the rarest query term, else 0."""
    return int(category == intended_category and rare_term is not None and rare_term in doc_tokens)


def build_pool() -> pd.DataFrame:
    """One row per (query, pooled document) with the columns the judge needs; rel is left empty."""
    index = InvertedIndex.load()
    articles = load_articles().set_index("doc_id")
    tokens = load_tokens()

    rows = []
    for q in load_queries().itertuples():
        rare = rarest_term(index, preprocess(q.query))
        for doc_id in pool_for_query(index, q.query):
            article = articles.loc[doc_id]
            rows.append({
                "query_id": q.query_id,
                "query": q.query,
                "doc_id": doc_id,
                "title": article["title"],
                "category": article["category"],
                # First 300 characters, with line breaks turned into spaces so each row stays on one line.
                "snippet": " ".join(article["text"][:300].split()),
                "suggested_rel": suggest_rel(set(tokens[doc_id]), article["category"], q.intended_category, rare),
                "rel": "",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pool = build_pool()
    # utf-8-sig adds a byte-order mark so Excel shows characters like £ correctly; pandas reads it fine.
    pool.to_csv(POOL_CSV, index=False, encoding="utf-8-sig")

    sizes = pool.groupby(["query_id", "query"]).agg(pool_size=("doc_id", "size"), suggested=("suggested_rel", "sum"))
    print("Pool size per query (suggested = rows with suggested_rel = 1):")
    print(sizes.to_string())
    print(f"\nTotal rows to judge: {len(pool)}  (min {sizes.pool_size.min()}, max {sizes.pool_size.max()}, "
          f"mean {sizes.pool_size.mean():.1f} per query)")
    print(f"Saved {POOL_CSV}")
