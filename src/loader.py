"""Step 2: load the raw BBC articles, remove duplicates, add a train/test split and save data/articles.csv.

Run from the project root:  python -m src.loader
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ARTICLES_CSV = "data/articles.csv"


def read_text_file(path: Path) -> str:
    """Read a file as UTF-8; if that fails (some BBC files are not valid UTF-8), read it as latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # latin-1 maps every byte to a character, so it never fails.
        return path.read_text(encoding="latin-1")


def split_title_and_text(content: str) -> tuple[str, str]:
    """Return (title, text): the title is the first non-empty line, the text is everything after it."""
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.strip():
            title = line.strip()
            text = "\n".join(lines[i + 1:]).strip()
            return title, text
    return "", ""  # empty file


def load_raw(data_dir: str = "data/bbc") -> pd.DataFrame:
    """Read every .txt file under data_dir/<category>/ into one row: category, title, text."""
    rows = []
    # Sort folders and files so the row order (and therefore doc_id) is the same on every run.
    for category_dir in sorted(Path(data_dir).iterdir()):
        if not category_dir.is_dir():
            continue  # skip README.TXT
        for path in sorted(category_dir.glob("*.txt")):
            title, text = split_title_and_text(read_text_file(path))
            rows.append({"category": category_dir.name, "title": title, "text": text})
    return pd.DataFrame(rows, columns=["category", "title", "text"])


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop articles whose title and text exactly match an earlier article; print counts before and after."""
    before = df["category"].value_counts().sort_index()
    deduped = df.drop_duplicates(subset=["title", "text"], keep="first").reset_index(drop=True)
    after = deduped["category"].value_counts().sort_index()

    counts = pd.DataFrame({"before": before, "after": after})
    counts["removed"] = counts["before"] - counts["after"]
    counts.loc["TOTAL"] = counts.sum()
    print("Articles per category before and after removing duplicates:")
    print(counts.to_string())
    return deduped


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add a split column ("train"/"test"): stratified 80/20 by category with random_state=42."""
    # Stratifying keeps each category's 80/20 proportion the same in train and test.
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=df["category"], random_state=42
    )
    df = df.copy()
    df.loc[train_idx, "split"] = "train"
    df.loc[test_idx, "split"] = "test"
    return df


def load_articles() -> pd.DataFrame:
    """Read data/articles.csv. Every later step uses this instead of re-reading the raw files."""
    return pd.read_csv(ARTICLES_CSV, encoding="utf-8")


if __name__ == "__main__":
    articles = remove_duplicates(load_raw())
    # doc_id is assigned after de-duplication so the ids run 0..N-1 with no gaps.
    articles.insert(0, "doc_id", range(len(articles)))
    articles = add_split(articles)
    articles = articles[["doc_id", "category", "title", "text", "split"]]
    articles.to_csv(ARTICLES_CSV, index=False, encoding="utf-8")

    print("\nSplit sizes per category:")
    split_counts = pd.crosstab(articles["category"], articles["split"], margins=True, margins_name="TOTAL")
    print(split_counts[["train", "test", "TOTAL"]].to_string())
    print(f"\nSaved {len(articles)} articles to {ARTICLES_CSV}")
