"""Tests for Step 4 (src/indexer.py), using the toy corpus from conftest.py."""

import math

import pytest

from src.indexer import InvertedIndex

D1, D2, D3 = 1, 2, 3


@pytest.fixture
def index(toy_docs) -> InvertedIndex:
    idx = InvertedIndex()
    idx.build(toy_docs)
    return idx


def test_df(index):
    assert index.df("apple") == 2
    assert index.df("unknown") == 0


def test_tf(index):
    assert index.postings["match"][D3] == 2


def test_stats(index):
    assert index.N == 3
    assert index.total_tokens == 13
    assert index.avgdl == pytest.approx(13 / 3)
    assert index.cf["match"] == 2


def test_candidates(index):
    assert index.candidates(["apple", "iphone"]) == {D1, D2}


def test_doc_norm_d1(index):
    # apple: df=2 -> log10(3/2); launch, new, iphone: df=1 -> log10(3); all tf=1 so (1 + log10 1) = 1
    expected = math.sqrt(math.log10(1.5) ** 2 + 3 * math.log10(3) ** 2)
    assert index.doc_norm[D1] == pytest.approx(expected)
    assert index.doc_norm[D1] == pytest.approx(0.845, abs=0.001)


def test_boolean_search(index):
    # Query strings go through preprocess(); these words are unchanged by stemming.
    assert index.boolean_and("share market") == {D2}
    assert index.boolean_and("share cricket") == set()
    assert index.boolean_or("cricket launch") == {D1, D3}
    assert index.boolean_not("launch share market", "new") == set()
    assert index.boolean_not("market", "cricket") == {D2}


def test_save_and_load(index, tmp_path):
    path = str(tmp_path / "index.pkl")
    index.save(path)
    loaded = InvertedIndex.load(path)
    assert loaded.postings == index.postings
    assert loaded.avgdl == index.avgdl
    assert loaded.doc_norm == index.doc_norm
