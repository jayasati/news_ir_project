"""Tests for Step 2 (src/loader.py). Run python -m src.loader first so data/articles.csv exists."""

import pandas as pd
import pytest

from src.loader import load_articles, remove_duplicates

CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}


@pytest.fixture(scope="module")
def articles() -> pd.DataFrame:
    return load_articles()


def test_columns(articles):
    assert list(articles.columns) == ["doc_id", "category", "title", "text", "split"]


def test_no_duplicates_remain(articles):
    assert not articles.duplicated(subset=["title", "text"]).any()
    assert not articles.duplicated(subset=["category", "title"]).any()
    assert not articles.duplicated(subset=["category", "text"]).any()


def test_near_copies_removed():
    words = [f"word{i}" for i in range(100)]
    base = " ".join(words)
    one_word_changed = " ".join(["fixed"] + words[1:])
    df = pd.DataFrame({
        "category": ["tech", "tech", "sport", "tech"],
        "title": ["Original", "Retitled", "Other sport", "Follow-up"],
        # Row 1 changes one word of 100 (Jaccard 99/101 = 0.98); row 2 is the same words in another
        # category; row 3 shares only half the words.
        "text": [base, one_word_changed, base, " ".join(f"word{i}" for i in range(50, 150))],
    })
    out = remove_duplicates(df)
    assert list(out["title"]) == ["Original", "Other sport", "Follow-up"]


def test_remove_duplicates_keeps_first():
    df = pd.DataFrame({
        "category": ["tech", "tech", "sport", "tech", "sport", "tech"],
        "title": ["A", "A", "B", "A", "A", "A again"],
        "text": ["same", "same", "other", "updated version", "different story", "same"],
    })
    out = remove_duplicates(df)
    # Row 1 is an exact copy, row 3 has the same category + title, row 5 has the same category + text
    # (re-titled). Row 4 shares the title but is in another category, so it is kept.
    assert list(out["text"]) == ["same", "other", "different story"]


def test_doc_ids_unique_and_contiguous(articles):
    assert articles["doc_id"].is_unique
    assert sorted(articles["doc_id"]) == list(range(len(articles)))


def test_every_category_in_both_splits(articles):
    assert set(articles["split"]) == {"train", "test"}
    for split in ["train", "test"]:
        assert set(articles.loc[articles["split"] == split, "category"]) == CATEGORIES


def test_test_share_about_20_percent_per_category(articles):
    for category, group in articles.groupby("category"):
        share = (group["split"] == "test").mean()
        # Rounding to whole articles means the share can be off by at most about one article.
        assert abs(share - 0.2) < 0.01, f"{category}: test share {share:.3f}"
