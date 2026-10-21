"""Step 2: exploratory data analysis figures for the de-duplicated articles.

Run from the project root:  python -m src.eda
"""

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.loader import load_articles

FIGURES_DIR = "reports/figures"


def plot_class_distribution(df: pd.DataFrame, path: str) -> None:
    """Bar chart of articles per category, with train and test side by side."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="category", hue="split", hue_order=["train", "test"], ax=ax)
    for bars in ax.containers:
        ax.bar_label(bars)  # print the count on top of each bar
    ax.set_title("Articles per category (train vs test)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of articles")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_doc_length(df: pd.DataFrame, path: str) -> None:
    """Histogram of words per article (title + text)."""
    # A rough raw length for EDA only: count whitespace-separated words. This is not used for retrieval;
    # all retrieval and classification text goes through preprocess() in src/preprocess.py.
    lengths = (df["title"] + " " + df["text"]).str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(lengths.median(), color="darkred", linestyle="--", label=f"median = {lengths.median():.0f}")
    ax.set_title("Words per article")
    ax.set_xlabel("Words (whitespace-separated)")
    ax.set_ylabel("Number of articles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Words per article: min {lengths.min()}, median {lengths.median():.0f}, "
          f"mean {lengths.mean():.0f}, max {lengths.max()}")


if __name__ == "__main__":
    articles = load_articles()
    plot_class_distribution(articles, f"{FIGURES_DIR}/class_distribution.png")
    plot_doc_length(articles, f"{FIGURES_DIR}/doc_length.png")
    print(f"Saved {FIGURES_DIR}/class_distribution.png and {FIGURES_DIR}/doc_length.png")
