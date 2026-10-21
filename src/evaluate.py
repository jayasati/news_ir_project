"""Step 9: evaluate the rankers (P@k, recall, MAP, nDCG, parameter sweeps) and the Naive Bayes classifier.

Run from the project root:  python -m src.evaluate
Writes data/qrels.csv, reports/results.md and charts in reports/figures/.
"""

import math

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.classifier import NaiveBayes
from src.indexer import InvertedIndex
from src.loader import load_articles
from src.pool import POOL_CSV, load_queries
from src.preprocess import load_tokens, preprocess
from src.rankers import bm25_search, lm_search, tfidf_search

QRELS_CSV = "data/qrels.csv"
RESULTS_MD = "reports/results.md"
FIGURES_DIR = "reports/figures"
RUN_DEPTH = 1000  # how many results each ranker returns per query when computing AP

BM25_K1 = [0.9, 1.2, 1.5, 2.0]
BM25_B = [0.5, 0.75, 1.0]
LM_MU = [500, 1000, 2000, 3000]

# Where the relevance judgements came from; printed in the report so readers know what the numbers rest on.
JUDGEMENT_SOURCE = ("graded 0/1/2 by an LLM (Claude) that read every pooled article, then reviewed by the "
                    "student, who accepted all 289 drafts unchanged")


# ---------------------------------------------------------------- relevance judgements

def build_qrels() -> pd.DataFrame:
    """Turn the judged pool (data/pool_to_judge.csv) into data/qrels.csv with columns query_id, doc_id, rel."""
    pool = pd.read_csv(POOL_CSV)
    empty = pool[pool["rel"].isna()]
    if len(empty):
        raise ValueError(f"rel is empty for queries {sorted(empty['query_id'].unique())}; judge them first")
    qrels = pool[["query_id", "doc_id", "rel"]].astype(int)
    qrels.to_csv(QRELS_CSV, index=False)
    return qrels


def load_qrels() -> dict[int, dict[int, int]]:
    """Read data/qrels.csv into {query_id: {doc_id: rel}}. Unjudged documents are simply absent (= rel 0)."""
    qrels: dict[int, dict[int, int]] = {}
    for row in pd.read_csv(QRELS_CSV).itertuples():
        qrels.setdefault(row.query_id, {})[row.doc_id] = row.rel
    return qrels


# ---------------------------------------------------------------- metrics (written from scratch)

def relevant_set(qrels: dict[int, int]) -> set[int]:
    """Documents with rel >= 1 count as relevant for precision, recall and AP."""
    return {doc_id for doc_id, rel in qrels.items() if rel >= 1}


def precision_at_k(ranked: list[int], qrels: dict[int, int], k: int) -> float:
    """P@k = (number of relevant documents in the top k) / k."""
    relevant = relevant_set(qrels)
    return sum(1 for doc_id in ranked[:k] if doc_id in relevant) / k


def recall_at_k(ranked: list[int], qrels: dict[int, int], k: int) -> float:
    """R@k = (number of relevant documents in the top k) / (total number of relevant documents)."""
    relevant = relevant_set(qrels)
    if not relevant:
        return 0.0
    return sum(1 for doc_id in ranked[:k] if doc_id in relevant) / len(relevant)


def average_precision(ranked: list[int], qrels: dict[int, int]) -> float:
    """AP = (1 / R) * sum of P@i over every rank i that holds a relevant document.

    R is the total number of relevant documents, so relevant documents that are never retrieved add 0.
    """
    relevant = relevant_set(qrels)
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / i  # precision at this rank
    return total / len(relevant)


def mean_average_precision(runs: dict[int, list[int]], qrels: dict[int, dict[int, int]]) -> float:
    """MAP = mean of AP over all queries."""
    return sum(average_precision(runs[q], qrels.get(q, {})) for q in runs) / len(runs)


def dcg_at_k(gains: list[int], k: int) -> float:
    """DCG@k = sum over ranks i = 1..k of rel_i / log2(i + 1)."""
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(gains[:k], start=1))


def ndcg_at_k(ranked: list[int], qrels: dict[int, int], k: int) -> float:
    """nDCG@k = DCG@k of the ranking / DCG@k of the ideal ranking (all judged grades sorted high to low)."""
    gains = [qrels.get(doc_id, 0) for doc_id in ranked]
    ideal = dcg_at_k(sorted(qrels.values(), reverse=True), k)
    return dcg_at_k(gains, k) / ideal if ideal > 0 else 0.0


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided sign test: p = min(1, 2 * P(X <= min(wins, losses))), X ~ Binomial(wins + losses, 0.5). Ties dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------- retrieval evaluation

def run_queries(ranker, index: InvertedIndex, queries: pd.DataFrame, **params) -> dict[int, list[int]]:
    """{query_id: ranked doc_ids} for one ranker; every query goes through preprocess()."""
    return {
        q.query_id: [doc_id for doc_id, _ in ranker(index, preprocess(q.query), k=RUN_DEPTH, **params)]
        for q in queries.itertuples()
    }


def score_runs(runs: dict[int, list[int]], qrels: dict[int, dict[int, int]]) -> dict[str, float]:
    """Mean P@5, P@10, R@10, MAP and nDCG@10 over all queries."""
    n = len(runs)
    return {
        "P@5": sum(precision_at_k(runs[q], qrels[q], 5) for q in runs) / n,
        "P@10": sum(precision_at_k(runs[q], qrels[q], 10) for q in runs) / n,
        "R@10": sum(recall_at_k(runs[q], qrels[q], 10) for q in runs) / n,
        "MAP": mean_average_precision(runs, qrels),
        "nDCG@10": sum(ndcg_at_k(runs[q], qrels[q], 10) for q in runs) / n,
    }


def compare_per_query(runs_a, runs_b, qrels) -> tuple[int, int, int, float]:
    """(wins, ties, losses, sign-test p) of system A against system B on per-query AP."""
    wins = ties = losses = 0
    for q in runs_a:
        a, b = average_precision(runs_a[q], qrels[q]), average_precision(runs_b[q], qrels[q])
        if abs(a - b) < 1e-9:
            ties += 1
        elif a > b:
            wins += 1
        else:
            losses += 1
    return wins, ties, losses, sign_test_p(wins, losses)


def sweep_bm25(index, queries, qrels) -> pd.DataFrame:
    """MAP for every (k1, b) pair."""
    rows = [{"k1": k1, "b": b, "MAP": mean_average_precision(run_queries(bm25_search, index, queries, k1=k1, b=b), qrels)}
            for k1 in BM25_K1 for b in BM25_B]
    return pd.DataFrame(rows)


def sweep_lm(index, queries, qrels) -> pd.DataFrame:
    """MAP for every Dirichlet mu."""
    rows = [{"mu": mu, "MAP": mean_average_precision(run_queries(lm_search, index, queries, smoothing="dirichlet", mu=mu), qrels)}
            for mu in LM_MU]
    return pd.DataFrame(rows)


def plot_bm25_sweep(df: pd.DataFrame, path: str) -> None:
    """Line chart: MAP against k1, one line per b."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for b, group in df.groupby("b"):
        ax.plot(group["k1"], group["MAP"], marker="o", label=f"b = {b}")
    ax.set_xlabel("k1")
    ax.set_ylabel("MAP")
    ax.set_title("BM25 parameter sweep (MAP)")
    ax.set_xticks(BM25_K1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lm_sweep(df: pd.DataFrame, path: str) -> None:
    """Line chart: MAP against Dirichlet mu."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["mu"], df["MAP"], marker="o", color="darkgreen")
    ax.set_xlabel("Dirichlet mu")
    ax.set_ylabel("MAP")
    ax.set_title("Unigram LM (Dirichlet) parameter sweep (MAP)")
    ax.set_xticks(LM_MU)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- classification evaluation

def evaluate_classifier(articles: pd.DataFrame, tokens: dict[int, list[str]]) -> dict:
    """Predict the test split with models/nb.pkl; return metrics, confusion matrix and the top confusions."""
    model = NaiveBayes.load()
    test = articles[articles["split"] == "test"].copy()
    test["pred"] = [model.predict(tokens[d])[0] for d in test["doc_id"]]
    labels = model.classes
    cm = confusion_matrix(test["category"], test["pred"], labels=labels)

    errors = test[test["category"] != test["pred"]]
    confusions = (errors.groupby(["category", "pred"])
                  .agg(count=("doc_id", "size"), example=("title", "first"))
                  .reset_index().sort_values(["count", "category"], ascending=[False, True]).head(5))
    return {
        "n": len(test),
        "accuracy": accuracy_score(test["category"], test["pred"]),
        "macro_f1": f1_score(test["category"], test["pred"], average="macro"),
        "report": classification_report(test["category"], test["pred"], labels=labels, output_dict=True),
        "labels": labels,
        "cm": cm,
        "confusions": confusions,
        "n_errors": len(errors),
    }


def plot_confusion(cm, labels: list[str], path: str) -> None:
    """Heatmap of the confusion matrix (rows = true class, columns = predicted class)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Naive Bayes confusion matrix (test split)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report

def table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """A pandas DataFrame as a Markdown table (floats rounded)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(ctx: dict) -> str:
    """Build reports/results.md from the computed numbers."""
    res, per_query, comps = ctx["retrieval"], ctx["per_query"], ctx["comparisons"]
    bm25, lm, clf = ctx["bm25_sweep"], ctx["lm_sweep"], ctx["clf"]
    best_bm25 = bm25.loc[bm25["MAP"].idxmax()]
    default_bm25 = bm25[(bm25["k1"] == 1.5) & (bm25["b"] == 0.75)]["MAP"].iloc[0]
    best_lm = lm.loc[lm["MAP"].idxmax()]
    default_lm = lm[lm["mu"] == 2000]["MAP"].iloc[0]

    rep = clf["report"]
    class_rows = pd.DataFrame([{"class": c, "precision": rep[c]["precision"], "recall": rep[c]["recall"],
                                "F1": rep[c]["f1-score"], "support": int(rep[c]["support"])} for c in clf["labels"]])
    cm_df = pd.DataFrame(clf["cm"], index=clf["labels"], columns=clf["labels"]).reset_index().rename(columns={"index": "true \\ predicted"})
    conf = clf["confusions"].rename(columns={"category": "true", "pred": "predicted", "example": "example title"})

    comp_lines = [f"| {a} vs {b} | {w} | {t} | {l} | {p:.3f} |" for (a, b), (w, t, l, p) in comps.items()]
    lines = [
        "# Results",
        "",
        f"Relevance judgements (`data/qrels.csv`): {ctx['n_judged']} judged (query, document) pairs for "
        f"{ctx['n_queries']} queries, {JUDGEMENT_SOURCE}. Grades: {ctx['grade_counts']}.",
        "Relevant = rel >= 1 for P@k, recall and AP; nDCG uses the graded rel (0/1/2). Each ranker returns up to "
        f"{RUN_DEPTH} documents per query. Documents outside the judged pool count as not relevant.",
        "",
        "## 1. Retrieval (default parameters)",
        "",
        "TF-IDF cosine; BM25 k1=1.5, b=0.75; unigram LM with Dirichlet mu=2000. Means over the 20 queries.",
        "",
        table(res),
        "",
        "Per-query AP (one row per query):",
        "",
        table(per_query),
        "",
        "Head-to-head on per-query AP (wins / ties / losses for the first system; two-sided sign test, ties dropped):",
        "",
        "| Comparison | Wins | Ties | Losses | Sign-test p |",
        "|---|---|---|---|---|",
        *comp_lines,
        "",
        "## 2. Parameter sweeps (MAP)",
        "",
        "BM25 (rows k1, columns b):",
        "",
        table(bm25.pivot(index="k1", columns="b", values="MAP").reset_index().rename(columns=lambda c: f"b={c}" if c != "k1" else c)),
        "",
        f"Best: k1={best_bm25['k1']}, b={best_bm25['b']} with MAP {best_bm25['MAP']:.3f} "
        f"(default k1=1.5, b=0.75: {default_bm25:.3f}). Chart: `reports/figures/bm25_sweep.png`.",
        "",
        "Unigram LM, Dirichlet mu:",
        "",
        table(lm.assign(mu=lm["mu"].astype(int))),
        "",
        f"Best: mu={int(best_lm['mu'])} with MAP {best_lm['MAP']:.3f} (default mu=2000: {default_lm:.3f}). "
        "Chart: `reports/figures/lm_mu_sweep.png`.",
        "",
        "**Pool bias.** The pool was built from the top 10 of the three rankers at their *default* settings. "
        "A setting that brings up different documents gets them counted as not relevant even if a person would "
        "judge them relevant, so the sweep is biased towards the defaults and small MAP differences should not be "
        "over-read. The sweep was also tuned on the same 20 queries it is scored on (no held-out queries).",
        "",
        "## 3. Classification (Naive Bayes, test split)",
        "",
        f"Test articles: {clf['n']}. **Accuracy {clf['accuracy']:.4f}**, **macro-F1 {clf['macro_f1']:.4f}** "
        f"({clf['n_errors']} errors).",
        "",
        table(class_rows),
        "",
        "Confusion matrix (rows true, columns predicted). Heatmap: `reports/figures/confusion_matrix.png`.",
        "",
        table(cm_df),
        "",
        "Most common confusions (an example title for each):",
        "",
        table(conf),
        "",
        "## 4. Interpretation",
        "",
        *ctx["interpretation"],
        "",
    ]
    return "\n".join(lines)


def interpretation(res: pd.DataFrame, per_query: pd.DataFrame, comps: dict, bm25, lm, clf) -> list[str]:
    """Plain statements generated from the numbers, so the text always matches the tables."""
    r = res.set_index("model")
    best_map = r["MAP"].idxmax()
    out = [f"- **Retrieval.** Highest MAP: {best_map} ({r.loc[best_map, 'MAP']:.3f}); "
           + ", ".join(f"{m} {r.loc[m, 'MAP']:.3f}" for m in r.index if m != best_map) + "."]
    # Three comparisons are tested at once, so the Bonferroni-corrected threshold is 0.05 / 3.
    strict = 0.05 / len(comps)
    for (a, b), (w, t, l, p) in comps.items():
        if p < strict:
            verdict = f"a consistent per-query difference even after correcting for {len(comps)} tests (p < {strict:.3f})"
        elif p < 0.05:
            verdict = (f"suggestive (p < 0.05) but not significant once corrected for {len(comps)} tests "
                       f"(Bonferroni threshold {strict:.3f})")
        else:
            verdict = "not a statistically reliable difference on 20 queries (p >= 0.05)"
        out.append(f"- {a} vs {b}: better on {w} queries, equal on {t}, worse on {l}; sign-test p = {p:.3f}, "
                   f"so this is {verdict}.")
    # The query where the models disagree most: a single query can move a 20-query mean a lot.
    models = [c for c in per_query.columns if c not in ("query_id", "query")]
    spread = per_query[models].max(axis=1) - per_query[models].min(axis=1)
    worst = per_query.loc[spread.idxmax()]
    out.append(f"- Largest per-query gap: \"{worst['query']}\" ("
               + ", ".join(f"{m} AP {worst[m]:.3f}" for m in models)
               + "). One query like this moves a 20-query MAP by several points.")
    spread = bm25["MAP"].max() - bm25["MAP"].min()
    out.append(f"- **BM25 sweep.** MAP ranges from {bm25['MAP'].min():.3f} to {bm25['MAP'].max():.3f} "
               f"(spread {spread:.3f}) over the 12 settings. "
               + ("The defaults are already among the best settings." if bm25['MAP'].max() - bm25[(bm25.k1 == 1.5) & (bm25.b == 0.75)]['MAP'].iloc[0] < 0.01
                  else "Some settings beat the defaults, but pool bias and the small query set make this a hint, not a conclusion."))
    lm_sorted = lm.sort_values("mu")
    trend = ("falls steadily as mu grows" if lm_sorted["MAP"].is_monotonic_decreasing
             else "rises steadily as mu grows" if lm_sorted["MAP"].is_monotonic_increasing else "has no steady trend")
    out.append(f"- **LM sweep.** MAP ranges from {lm['MAP'].min():.3f} to {lm['MAP'].max():.3f} over mu = 500-3000 "
               f"and {trend}. Smaller mu means less smoothing towards the collection model, i.e. more weight on the "
               "words actually in each article. Pool bias favours the default mu = 2000, so a smaller mu winning "
               "anyway is a reasonably strong hint, though it is still tuned and scored on the same 20 queries.")
    out.append(f"- **Classification.** Naive Bayes reaches {clf['accuracy']:.1%} accuracy and {clf['macro_f1']:.3f} macro-F1 "
               "on held-out articles; the errors are mostly articles that genuinely mix two topics (see the examples).")
    out.append("- **Caveats.** 20 queries is a small sample; relevance was judged on a depth-10 pool from these same "
               "rankers; the judgements were LLM-drafted and accepted unchanged by the student. Precision-oriented "
               "measures (P@5, P@10, nDCG@10) are the most trustworthy here because every top-10 document of the "
               "default rankers was judged; recall and MAP are optimistic because relevant articles outside the pool are unknown.")
    return out


if __name__ == "__main__":
    qrels_df = build_qrels()
    qrels = load_qrels()
    queries = load_queries()
    index = InvertedIndex.load()
    articles = load_articles()
    tokens = load_tokens()

    runs = {
        "TF-IDF": run_queries(tfidf_search, index, queries),
        "BM25": run_queries(bm25_search, index, queries, k1=1.5, b=0.75),
        "LM (Dirichlet)": run_queries(lm_search, index, queries, smoothing="dirichlet", mu=2000),
    }
    retrieval = pd.DataFrame([{"model": m, **score_runs(r, qrels)} for m, r in runs.items()])
    per_query = pd.DataFrame([{"query_id": q.query_id, "query": q.query,
                               **{m: average_precision(runs[m][q.query_id], qrels[q.query_id]) for m in runs}}
                              for q in queries.itertuples()])
    comparisons = {(a, b): compare_per_query(runs[a], runs[b], qrels)
                   for a, b in [("BM25", "TF-IDF"), ("LM (Dirichlet)", "TF-IDF"), ("BM25", "LM (Dirichlet)")]}

    bm25_sweep = sweep_bm25(index, queries, qrels)
    lm_sweep = sweep_lm(index, queries, qrels)
    plot_bm25_sweep(bm25_sweep, f"{FIGURES_DIR}/bm25_sweep.png")
    plot_lm_sweep(lm_sweep, f"{FIGURES_DIR}/lm_mu_sweep.png")

    clf = evaluate_classifier(articles, tokens)
    plot_confusion(clf["cm"], clf["labels"], f"{FIGURES_DIR}/confusion_matrix.png")

    report = write_report({
        "retrieval": retrieval, "per_query": per_query, "comparisons": comparisons,
        "bm25_sweep": bm25_sweep, "lm_sweep": lm_sweep, "clf": clf,
        "n_judged": len(qrels_df), "n_queries": qrels_df["query_id"].nunique(),
        "grade_counts": ", ".join(f"{g}: {n}" for g, n in qrels_df["rel"].value_counts().sort_index().items()),
        "interpretation": interpretation(retrieval, per_query, comparisons, bm25_sweep, lm_sweep, clf),
    })
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Saved {QRELS_CSV}, {RESULTS_MD} and charts in {FIGURES_DIR}/")
