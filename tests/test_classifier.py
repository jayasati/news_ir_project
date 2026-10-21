"""Tests for Step 6 (src/classifier.py) using IIR Example 13.1 (Manning, Raghavan & Schütze).

The tokens are passed directly, without preprocess().
"""

import math

import pytest

from src.classifier import NaiveBayes

TRAIN_DOCS = [
    ["Chinese", "Beijing", "Chinese"],
    ["Chinese", "Chinese", "Shanghai"],
    ["Chinese", "Macao"],
    ["Tokyo", "Japan", "Chinese"],
]
TRAIN_LABELS = ["c", "c", "c", "j"]
TEST_DOC = ["Chinese", "Chinese", "Chinese", "Tokyo", "Japan"]


@pytest.fixture
def model() -> NaiveBayes:
    nb = NaiveBayes()
    nb.fit(TRAIN_DOCS, TRAIN_LABELS, alpha=1.0)
    return nb


def test_priors(model):
    assert math.exp(model.log_prior["c"]) == pytest.approx(3 / 4)
    assert math.exp(model.log_prior["j"]) == pytest.approx(1 / 4)


def test_likelihoods(model):
    # |V| = 6; class c has 8 tokens, class j has 3.
    assert math.exp(model.log_likelihood["c"]["Chinese"]) == pytest.approx(3 / 7)   # (5+1)/(8+6)
    assert math.exp(model.log_likelihood["j"]["Tokyo"]) == pytest.approx(2 / 9)     # (1+1)/(3+6)
    assert math.exp(model.log_likelihood["c"]["Tokyo"]) == pytest.approx(1 / 14)    # (0+1)/(8+6)


def test_predict_textbook_document(model):
    label, scores = model.predict(TEST_DOC)
    assert label == "c"
    # Book values: P(c|d) ∝ 3/4 · (3/7)^3 · (1/14)^2 ≈ 0.0003, P(j|d) ∝ 1/4 · (2/9)^3 · (2/9)^2 ≈ 0.0001
    assert math.exp(scores["c"]) == pytest.approx(0.75 * (3 / 7) ** 3 * (1 / 14) ** 2)
    assert math.exp(scores["j"]) == pytest.approx(0.25 * (2 / 9) ** 5)


def test_predict_proba_sums_to_one(model):
    proba = model.predict_proba(TEST_DOC)
    assert sum(proba.values()) == pytest.approx(1.0)
    assert proba["c"] > proba["j"]


def test_unknown_words_ignored(model):
    _, with_unknown = model.predict(TEST_DOC + ["zebra"])
    _, without = model.predict(TEST_DOC)
    assert with_unknown == without


def test_top_words_and_explain(model):
    assert model.top_words("j", 2)[0][0] in {"Tokyo", "Japan"}
    assert [t for t, _ in model.explain(TEST_DOC, "c")] == ["Chinese"]


def test_save_and_load(model, tmp_path):
    path = str(tmp_path / "nb.pkl")
    model.save(path)
    loaded = NaiveBayes.load(path)
    assert loaded.predict(TEST_DOC) == model.predict(TEST_DOC)
