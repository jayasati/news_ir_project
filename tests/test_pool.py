"""Tests for Step 8 (src/pool.py) helpers, using the toy corpus from conftest.py."""

import pandas as pd

from src.indexer import InvertedIndex
from src.pool import load_queries, rarest_term, suggest_rel

CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}


def test_rarest_term(toy_docs):
    index = InvertedIndex()
    index.build(toy_docs)
    assert rarest_term(index, ["apple", "iphone"]) == "iphone"  # df 1 beats df 2
    assert rarest_term(index, ["zebra"]) is None


def test_suggest_rel():
    assert suggest_rel({"elect", "labour"}, "politics", "politics", "labour") == 1
    assert suggest_rel({"elect", "labour"}, "business", "politics", "labour") == 0  # wrong category
    assert suggest_rel({"elect"}, "politics", "politics", "labour") == 0             # rare term missing
    assert suggest_rel({"elect"}, "politics", "politics", None) == 0


def test_queries_file():
    queries = load_queries()
    assert list(queries.columns) == ["query_id", "query", "intended_category"]
    assert len(queries) == 20
    assert queries["query_id"].is_unique
    assert queries["intended_category"].value_counts().to_dict() == {c: 4 for c in CATEGORIES}
    for q in queries["query"]:
        words = q.split()
        assert 2 <= len(words) <= 4
        assert not set(words) & CATEGORIES  # category names are not used as queries
