"""Tests for Step 3 (src/preprocess.py)."""

from src.preprocess import preprocess


def test_stopwords_removed_and_stemmed():
    assert preprocess("The elections were held") == ["elect", "held"]


def test_punctuation_and_digits_dropped():
    assert preprocess("Running, runs 2005!!") == ["run", "run"]


def test_query_and_document_share_stems():
    query = preprocess("elections")
    document = preprocess("Labour won the General Elections last May.")
    assert query == ["elect"]
    assert query[0] in document


def test_one_letter_tokens_dropped():
    # "u" and "k" come from "U.K." after punctuation is removed
    assert preprocess("U.K. x") == []
