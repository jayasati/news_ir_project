"""Step 3: the one and only text preprocessing function, used for both documents and queries.

Run from the project root:  python -m src.preprocess
"""

import pickle
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.loader import load_articles

TOKENS_PKL = "data/tokens.pkl"

# Load the stopword list and the stemmer once, when the module is imported, not on every call.
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)  # first run on a new machine
    STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

# A token is a run of letters a-z. Anything else (punctuation, digits, spaces) separates tokens.
TOKEN_PATTERN = re.compile(r"[a-z]+")


def preprocess(text: str) -> list[str]:
    """Turn raw text into a list of stems: lowercase -> [a-z]+ tokens -> drop stopwords and 1-letter tokens -> Porter stem."""
    tokens = TOKEN_PATTERN.findall(text.lower())
    # Stopwords are checked before stemming because the NLTK list contains whole words ("were", not "were" stemmed).
    kept = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return [STEMMER.stem(t) for t in kept]


def tokenize_all() -> dict[int, list[str]]:
    """Preprocess title + text of every article, keyed by doc_id, and save to data/tokens.pkl."""
    articles = load_articles()
    tokens = {
        int(doc_id): preprocess(f"{title} {text}")
        for doc_id, title, text in zip(articles["doc_id"], articles["title"], articles["text"])
    }
    with open(TOKENS_PKL, "wb") as f:
        pickle.dump(tokens, f)
    return tokens


def load_tokens() -> dict[int, list[str]]:
    """Read data/tokens.pkl. Later steps use this instead of re-tokenizing."""
    with open(TOKENS_PKL, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    tokens = tokenize_all()
    counts = Counter(t for doc in tokens.values() for t in doc)
    total = sum(counts.values())
    print(f"Saved tokens for {len(tokens)} articles to {TOKENS_PKL}")
    print(f"Vocabulary size (distinct stems): {len(counts)}")
    print(f"Total tokens: {total}")
    print(f"Average tokens per article: {total / len(tokens):.1f}")
    print("20 most frequent stems:")
    for stem, n in counts.most_common(20):
        print(f"  {stem:<10} {n}")
