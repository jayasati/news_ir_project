"""Tests for Step 9 metrics (src/evaluate.py) on a tiny ranking worked out by hand.

Ranking for query 1: [1, 2, 3, 4, 5]
Judgements for query 1: doc 1 = 2 (highly relevant), doc 2 = 0, doc 3 = 1, doc 6 = 1 (never retrieved).
Relevant (rel >= 1) = {1, 3, 6}, so R = 3.
"""

import math

import pytest

from src.evaluate import (average_precision, mean_average_precision, ndcg_at_k, precision_at_k, recall_at_k,
                          sign_test_p)

RANKED = [1, 2, 3, 4, 5]
QRELS = {1: 2, 2: 0, 3: 1, 6: 1}


def test_precision_at_k():
    assert precision_at_k(RANKED, QRELS, 1) == 1.0          # doc 1 relevant
    assert precision_at_k(RANKED, QRELS, 3) == pytest.approx(2 / 3)  # docs 1 and 3
    assert precision_at_k(RANKED, QRELS, 5) == pytest.approx(2 / 5)


def test_recall_at_k():
    assert recall_at_k(RANKED, QRELS, 1) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, QRELS, 5) == pytest.approx(2 / 3)  # doc 6 never retrieved
    assert recall_at_k(RANKED, {}, 5) == 0.0


def test_average_precision():
    # Relevant hits at ranks 1 and 3: P@1 = 1, P@3 = 2/3. Doc 6 is never retrieved and adds 0.
    # AP = (1 + 2/3) / 3 = 5/9
    assert average_precision(RANKED, QRELS) == pytest.approx(5 / 9)
    assert average_precision(RANKED, {2: 0}) == 0.0  # no relevant documents


def test_map():
    # Query 2: the only relevant document is ranked first, so AP = 1. MAP = (5/9 + 1) / 2 = 7/9
    runs = {1: RANKED, 2: [7, 8]}
    qrels = {1: QRELS, 2: {7: 1}}
    assert mean_average_precision(runs, qrels) == pytest.approx(7 / 9)


def test_ndcg():
    # DCG@5 = 2/log2(2) + 0 + 1/log2(4) = 2 + 0.5 = 2.5
    # Ideal grades sorted: [2, 1, 1, 0] -> IDCG@5 = 2/1 + 1/log2(3) + 1/log2(4) = 3.1309...
    idcg = 2 + 1 / math.log2(3) + 0.5
    assert ndcg_at_k(RANKED, QRELS, 5) == pytest.approx(2.5 / idcg)
    assert ndcg_at_k(RANKED, QRELS, 5) == pytest.approx(0.7985, abs=1e-4)
    assert ndcg_at_k([1, 3, 6], QRELS, 3) == pytest.approx(1.0)  # ideal order


def test_sign_test():
    # 8 wins, 2 losses: P(X <= 2) for Binomial(10, 0.5) = (1 + 10 + 45) / 1024 = 56/1024; two-sided = 112/1024
    assert sign_test_p(8, 2) == pytest.approx(112 / 1024)
    assert sign_test_p(5, 5) == 1.0
    assert sign_test_p(0, 0) == 1.0
