"""Step 4: the inverted index, plus Boolean AND / OR / NOT search.

Run from the project root:  python -m src.indexer
"""

import math
import pickle
from collections import Counter

from src.preprocess import load_tokens, preprocess

INDEX_PKL = "models/index.pkl"


class InvertedIndex:
    """Maps each term to the documents that contain it, plus the statistics the rankers need."""

    def __init__(self) -> None:
        self.postings: dict[str, dict[int, int]] = {}  # term -> {doc_id: tf (term frequency in that doc)}
        self.doc_len: dict[int, int] = {}               # doc_id -> number of tokens in the doc
        self.cf: dict[str, int] = {}                    # term -> collection frequency (total count in all docs)
        self.doc_norm: dict[int, float] = {}            # doc_id -> length of its TF-IDF vector (for cosine)
        self.N: int = 0                                 # number of documents
        self.total_tokens: int = 0                      # number of tokens in the whole collection
        self.avgdl: float = 0.0                         # average document length = total_tokens / N

    def build(self, docs: dict[int, list[str]]) -> None:
        """Fill the index from {doc_id: [already preprocessed tokens]}."""
        for doc_id, tokens in docs.items():
            self.doc_len[doc_id] = len(tokens)
            # Counter gives the tf of every term in this doc in one pass.
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, {})[doc_id] = tf
                self.cf[term] = self.cf.get(term, 0) + tf
        self.N = len(docs)
        self.total_tokens = sum(self.doc_len.values())
        self.avgdl = self.total_tokens / self.N if self.N else 0.0
        self._compute_doc_norms()

    def _compute_doc_norms(self) -> None:
        """doc_norm[d] = sqrt( sum over terms t in d of w(t,d)^2 ),  w(t,d) = (1 + log10 tf) * log10(N / df)."""
        squares = {doc_id: 0.0 for doc_id in self.doc_len}
        # Walk term by term: df is the same for every doc in one postings list.
        for term, plist in self.postings.items():
            idf = math.log10(self.N / len(plist))
            for doc_id, tf in plist.items():
                weight = (1 + math.log10(tf)) * idf
                squares[doc_id] += weight ** 2
        self.doc_norm = {doc_id: math.sqrt(s) for doc_id, s in squares.items()}

    def df(self, term: str) -> int:
        """Document frequency: the number of documents that contain the term (0 if unseen)."""
        return len(self.postings.get(term, {}))

    def candidates(self, q_terms: list[str]) -> set[int]:
        """Union of the postings of the query terms: every doc containing at least one of them."""
        result: set[int] = set()
        for term in q_terms:
            result |= self.postings.get(term, {}).keys()
        return result

    def _docs_with(self, term: str) -> set[int]:
        """The set of doc_ids whose postings contain the term."""
        return set(self.postings.get(term, {}))

    def boolean_and(self, query: str) -> set[int]:
        """Docs that contain ALL query terms (intersection of postings)."""
        terms = preprocess(query)
        if not terms:
            return set()
        result = self._docs_with(terms[0])
        for term in terms[1:]:
            result &= self._docs_with(term)
        return result

    def boolean_or(self, query: str) -> set[int]:
        """Docs that contain ANY query term (union of postings)."""
        return self.candidates(preprocess(query))

    def boolean_not(self, query_a: str, query_b: str) -> set[int]:
        """Docs matching query_a (AND of its terms) that contain NONE of query_b's terms: A AND NOT B."""
        return self.boolean_and(query_a) - self.boolean_or(query_b)

    def save(self, path: str = INDEX_PKL) -> None:
        """Pickle the index's attributes (a plain dict) to path."""
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str = INDEX_PKL) -> "InvertedIndex":
        """Create an InvertedIndex from a file written by save()."""
        index = cls()
        with open(path, "rb") as f:
            index.__dict__.update(pickle.load(f))
        return index


if __name__ == "__main__":
    # Index ALL articles (train and test): search should be able to find any article.
    index = InvertedIndex()
    index.build(load_tokens())
    index.save(INDEX_PKL)

    print(f"Saved index to {INDEX_PKL}")
    print(f"N (documents): {index.N}")
    print(f"Vocabulary size: {len(index.postings)}")
    print(f"Total tokens: {index.total_tokens}")
    print(f"avgdl: {index.avgdl:.2f}")
    print(f'df("elect"): {index.df("elect")}')
    first_10 = sorted(index.postings["elect"].items())[:10]
    print(f'First 10 postings for "elect" (doc_id, tf): {first_10}')
