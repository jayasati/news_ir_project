"""Step 10: Streamlit app - search BBC articles, classify new text, and show the evaluation results.

Run from the project root:  streamlit run app.py
"""

import html
import os
import re

import pandas as pd
import streamlit as st

from src.classifier import NaiveBayes
from src.indexer import InvertedIndex
from src.loader import load_articles
from src.preprocess import load_tokens, preprocess
from src.rankers import RANKERS, search

RESULTS_MD = "reports/results.md"
FIGURES = [
    ("BM25 parameter sweep", "reports/figures/bm25_sweep.png"),
    ("Unigram LM (Dirichlet) parameter sweep", "reports/figures/lm_mu_sweep.png"),
    ("Naive Bayes confusion matrix", "reports/figures/confusion_matrix.png"),
]
CATEGORY_COLOURS = {
    "business": "#1f77b4", "entertainment": "#9467bd", "politics": "#d62728",
    "sport": "#2ca02c", "tech": "#ff7f0e",
}
SNIPPET_CHARS = 250


# ---------------------------------------------------------------- load everything once

@st.cache_resource
def load_resources() -> tuple[InvertedIndex, NaiveBayes, pd.DataFrame]:
    """Load the index, the classifier and the articles once, and predict a category for every article."""
    index = InvertedIndex.load()
    model = NaiveBayes.load()
    articles = load_articles().set_index("doc_id")
    tokens = load_tokens()  # the same preprocessed tokens the model was trained on
    articles["predicted"] = [model.predict(tokens[d])[0] for d in articles.index]
    return index, model, articles


# ---------------------------------------------------------------- small display helpers

def badge(category: str) -> str:
    """A coloured HTML label for a category."""
    colour = CATEGORY_COLOURS.get(category, "#555555")
    return (f'<span style="background:{colour};color:white;padding:2px 8px;border-radius:10px;'
            f'font-size:0.8em">{category}</span>')


def snippet_with_highlights(text: str, query_stems: set[str]) -> str:
    """About 250 characters of text around the first query word, with query words wrapped in <mark>.

    A word is highlighted if preprocess() turns it into one of the query's stems, so "elections" matches the
    query "election" exactly as the search engine does.
    """
    text = " ".join(text.split())  # one line: paragraph breaks would break the HTML layout
    words = list(re.finditer(r"[A-Za-z]+", text))
    first_hit = next((w.start() for w in words if set(preprocess(w.group())) & query_stems), 0)
    start = max(0, first_hit - 60)
    start = text.rfind(" ", 0, start) + 1 if start > 0 else 0  # begin on a word boundary
    piece = text[start:start + SNIPPET_CHARS]

    # Rebuild the snippet word by word: escape everything (it is shown as HTML), mark the matching words.
    out, last = [], 0
    for w in re.finditer(r"[A-Za-z]+", piece):
        out.append(html.escape(piece[last:w.start()]))
        word = html.escape(w.group())
        out.append(f"<mark>{word}</mark>" if set(preprocess(w.group())) & query_stems else word)
        last = w.end()
    out.append(html.escape(piece[last:]))
    prefix = "... " if start > 0 else ""
    suffix = " ..." if start + SNIPPET_CHARS < len(text) else ""
    return prefix + "".join(out) + suffix


# ---------------------------------------------------------------- tab 1: search

def search_tab(index: InvertedIndex, articles: pd.DataFrame) -> None:
    """Query box, model choice, number of results and a filter on the predicted category."""
    query = st.text_input("Search query", placeholder="e.g. interest rates bank", key="query")
    col1, col2 = st.columns(2)
    model = col1.selectbox("Ranking model", ["BM25", "TF-IDF", "Unigram LM"], key="model")
    n_results = col2.slider("Number of results", 5, 50, 10, key="n_results")
    categories = sorted(CATEGORY_COLOURS)
    chosen = st.multiselect("Show predicted categories", categories, default=categories, key="categories")

    if not query.strip():
        st.info("Type a query above, for example \"interest rates bank\" or \"chelsea premiership\".")
        return
    query_stems = set(preprocess(query))
    if not query_stems:
        st.warning("Your query has no searchable words left after preprocessing "
                   "(it contained only stopwords, numbers or punctuation). Try more specific words.")
        return

    # Rank deeply, then apply the category filter, then keep the top n - so filtering never hides results
    # that would otherwise have made the list.
    ranked = search(index, query, model, k=1000)
    ranked = [(d, s) for d, s in ranked if articles.loc[d, "predicted"] in chosen][:n_results]
    if not ranked:
        st.warning("No articles match this query" + (" in the selected categories." if len(chosen) < 5 else ".")
                   + " Try other words or select more categories.")
        return

    st.caption(f"{len(ranked)} results with {model}. "
               + ("Unigram LM scores are log-probabilities: closer to 0 is better." if model == "Unigram LM" else ""))
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        art = articles.loc[doc_id]
        mismatch = art["predicted"] != art["category"]
        true_label = f"true: {art['category']}" + (" &#9888; differs from prediction" if mismatch else "")
        st.markdown(
            f"**{rank}. {html.escape(art['title'])}** &nbsp; score {score:.4f}<br>"
            f"{badge(art['predicted'])} <span style='font-size:0.8em'>predicted &middot; {true_label}</span><br>"
            f"<span style='font-size:0.9em'>{snippet_with_highlights(art['text'], query_stems)}</span>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- tab 2: classify an article

def classify_tab(model: NaiveBayes) -> None:
    """Paste any text; show the predicted category, the class probabilities and the deciding words."""
    text = st.text_area("Paste an article or a few sentences", height=200, key="article_text")
    if not st.button("Classify", key="classify"):
        return
    tokens = preprocess(text)
    known = [t for t in tokens if t in model.vocab]
    if not known:
        st.warning("None of these words appear in the training data, so the model cannot classify the text. "
                   "Try a longer piece of news text.")
        return

    label, scores = model.predict(tokens)
    runner_up = max((c for c in model.classes if c != label), key=scores.get)
    proba = model.predict_proba(tokens)
    st.markdown(f"Predicted category: {badge(label)} &nbsp; (runner-up: {runner_up})", unsafe_allow_html=True)
    st.caption(f"{len(known)} of {len(tokens)} words after preprocessing are in the model's vocabulary.")

    st.subheader("Probability per category")
    st.bar_chart(pd.Series(proba, name="probability"))

    st.subheader(f"Words that pushed the decision towards {label} rather than {runner_up}")
    words = model.explain(tokens, label, n=10)
    if words:
        st.bar_chart(pd.DataFrame(words, columns=["word (stem)", "log-likelihood ratio"]).set_index("word (stem)"))
    else:
        st.write("No single word clearly favours this category.")


# ---------------------------------------------------------------- tab 3: evaluation

def evaluation_tab() -> None:
    """Show reports/results.md (tables and interpretation) and the saved charts."""
    if not os.path.exists(RESULTS_MD):
        st.info("No results yet. Run `python -m src.evaluate` first.")
        return
    with open(RESULTS_MD, encoding="utf-8") as f:
        st.markdown(f.read())
    st.header("Charts")
    for caption, path in FIGURES:
        if os.path.exists(path):
            st.image(path, caption=caption)


# ---------------------------------------------------------------- page layout

st.set_page_config(page_title="BBC News Search", layout="wide")
st.title("Intelligent News Retrieval and Classification")
st.caption("2,058 BBC News articles (2004-2005). Search ranks articles with TF-IDF, BM25 or a unigram language "
           "model; categories are predicted by a Naive Bayes classifier.")

index, model, articles = load_resources()
tab_search, tab_classify, tab_eval = st.tabs(["Search", "Classify an article", "Evaluation"])
with tab_search:
    search_tab(index, articles)
with tab_classify:
    classify_tab(model)
with tab_eval:
    evaluation_tab()
