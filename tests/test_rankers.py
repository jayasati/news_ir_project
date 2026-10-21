"""Tests for Step 5 (src/rankers.py), using the toy corpus from conftest.py and the query apple iphone."""

import pytest

from src.indexer import InvertedIndex
from src.rankers import RANKERS, bm25_search, lm_search, tfidf_search

D1, D2, D3 = 1, 2, 3
QUERY = ["apple", "iphone"]  # already processed tokens, like the toy corpus


@pytest.fixture
def index(toy_docs) -> InvertedIndex:
    idx = InvertedIndex()
    idx.build(toy_docs)
    return idx


def test_tfidf(index):
    scores = dict(tfidf_search(index, QUERY))
    assert scores[D1] == pytest.approx(0.31, abs=0.005)
    assert scores[D2] == pytest.approx(0.04, abs=0.005)


def test_bm25(index):
    scores = dict(bm25_search(index, QUERY, k1=1.2, b=0.75))
    assert scores[D1] == pytest.approx(1.50, abs=0.005)
    assert scores[D2] == pytest.approx(0.49, abs=0.005)


def test_lm_jelinek_mercer(index):
    scores = dict(lm_search(index, QUERY, smoothing="jm", lam=0.1))
    assert scores[D1] == pytest.approx(-2.88, abs=0.005)
    assert scores[D2] == pytest.approx(-6.29, abs=0.005)


def test_lm_dirichlet_ranks_d1_first(index):
    results = lm_search(index, QUERY)  # default: dirichlet, mu=2000
    assert [doc_id for doc_id, _ in results] == [D1, D2]


def test_d3_never_returned(index):
    for ranker in RANKERS.values():
        doc_ids = [doc_id for doc_id, _ in ranker(index, QUERY)]
        assert D3 not in doc_ids
        assert doc_ids[0] == D1  # D1 has both query words, so it is always first


def test_unknown_terms_and_k(index):
    for ranker in RANKERS.values():
        assert ranker(index, ["zebra"]) == []
        assert len(ranker(index, QUERY + ["zebra"], k=1)) == 1
