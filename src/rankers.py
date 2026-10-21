"""Step 5: three ranking models over the inverted index: TF-IDF cosine, BM25 and the unigram language model.

Each ranker takes (index, q_terms, k) where q_terms is the output of preprocess(query), scores only the
candidate documents (those containing at least one query term) and returns [(doc_id, score), ...] best first.
"""

import math
from collections import Counter

from src.indexer import InvertedIndex
from src.preprocess import preprocess


def top_k(scores: dict[int, float], k: int) -> list[tuple[int, float]]:
    """Sort by score (highest first), breaking ties by doc_id so results are always in the same order."""
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:k]


def tfidf_search(index: InvertedIndex, q_terms: list[str], k: int = 10) -> list[tuple[int, float]]:
    """TF-IDF cosine similarity.

    score(q, d) = Σ_t w(t,q) · w(t,d) / |d|
        w(t,q) = (1 + log10 qtf) · idf(t)      qtf = count of t in the query
        w(t,d) = (1 + log10 tf)  · idf(t)      tf  = count of t in document d
        idf(t) = log10(N / df(t))
        |d|    = index.doc_norm[d] = sqrt(Σ_t w(t,d)²)   (precomputed in Step 4)

    Full cosine would also divide by the query length |q|. |q| is the same for every document of one query,
    so leaving it out does not change the ranking.
    """
    scores: dict[int, float] = {}
    for term, qtf in Counter(q_terms).items():
        df = index.df(term)
        if df == 0:
            continue  # term not in the collection: it matches nothing
        idf = math.log10(index.N / df)
        w_q = (1 + math.log10(qtf)) * idf
        for doc_id, tf in index.postings[term].items():
            w_d = (1 + math.log10(tf)) * idf
            scores[doc_id] = scores.get(doc_id, 0.0) + w_q * w_d
    # Divide each dot product by the document's vector length (doc_norm is never 0 for a doc with a matching term
    # unless every term in it appears in all N docs; guard anyway).
    scores = {d: s / index.doc_norm[d] for d, s in scores.items() if index.doc_norm[d] > 0}
    return top_k(scores, k)


def bm25_search(index: InvertedIndex, q_terms: list[str], k: int = 10,
                k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
    """Okapi BM25.

    score(q, d) = Σ_{t in q} idf(t) · tf · (k1 + 1) / (tf + k1 · (1 - b + b · |d| / avgdl))
        idf(t) = ln((N - df + 0.5) / (df + 0.5) + 1)
        tf = count of t in d,  |d| = doc_len[d],  avgdl = average document length
        k1 controls how fast repeated words stop adding score (term-frequency saturation),
        b  controls how much long documents are penalised (length normalisation).
    """
    scores: dict[int, float] = {}
    for term in q_terms:
        df = index.df(term)
        if df == 0:
            continue
        idf = math.log((index.N - df + 0.5) / (df + 0.5) + 1)
        for doc_id, tf in index.postings[term].items():
            length_norm = 1 - b + b * index.doc_len[doc_id] / index.avgdl
            term_score = idf * tf * (k1 + 1) / (tf + k1 * length_norm)
            scores[doc_id] = scores.get(doc_id, 0.0) + term_score
    return top_k(scores, k)


def lm_search(index: InvertedIndex, q_terms: list[str], k: int = 10,
              smoothing: str = "dirichlet", mu: float = 2000, lam: float = 0.1) -> list[tuple[int, float]]:
    """Query-likelihood unigram language model: rank d by log P(q | d) = Σ_{t in q} ln P(t | d).

    P(t | C) = cf(t) / total_tokens                          (collection / background model)
    Dirichlet:      P(t | d) = (tf + mu · P(t|C)) / (|d| + mu)
    Jelinek-Mercer: P(t | d) = (1 - lam) · tf / |d| + lam · P(t|C)

    Smoothing gives a query word missing from d a small non-zero probability, so ln P(t|d) is never ln 0.
    Query terms that never occur in the collection (cf = 0) are skipped: they would have probability 0
    in every document. Scores are negative; the closest to 0 is the best.
    """
    known_terms = [t for t in q_terms if index.cf.get(t, 0) > 0]
    scores: dict[int, float] = {}
    for doc_id in index.candidates(known_terms):
        doc_len = index.doc_len[doc_id]
        log_prob = 0.0
        for term in known_terms:
            tf = index.postings[term].get(doc_id, 0)
            p_collection = index.cf[term] / index.total_tokens
            if smoothing == "dirichlet":
                p = (tf + mu * p_collection) / (doc_len + mu)
            elif smoothing == "jm":
                p = (1 - lam) * tf / doc_len + lam * p_collection
            else:
                raise ValueError(f"unknown smoothing: {smoothing}")
            log_prob += math.log(p)
        scores[doc_id] = log_prob
    return top_k(scores, k)


# Used by the app (Step 10) to list the models by name.
RANKERS = {
    "TF-IDF": tfidf_search,
    "BM25": bm25_search,
    "Unigram LM": lm_search,
}


def search(index: InvertedIndex, query: str, model: str, k: int = 10) -> list[tuple[int, float]]:
    """Preprocess a raw query string and rank it with the model named in RANKERS (default parameters)."""
    return RANKERS[model](index, preprocess(query), k)
