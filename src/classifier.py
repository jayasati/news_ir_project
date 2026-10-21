"""Step 6: multinomial Naïve Bayes text classifier, written from scratch.

Run from the project root:  python -m src.classifier
"""

import math
import pickle
from collections import Counter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.loader import load_articles
from src.preprocess import load_tokens

NB_PKL = "models/nb.pkl"


class NaiveBayes:
    """Multinomial Naïve Bayes: a document is a bag of words drawn independently from its class."""

    def __init__(self) -> None:
        self.classes: list[str] = []
        self.vocab: set[str] = set()                           # every word seen in training
        self.log_prior: dict[str, float] = {}                  # c -> log P(c)
        self.log_likelihood: dict[str, dict[str, float]] = {}  # c -> {word: log P(word | c)}
        self.alpha: float = 1.0

    def fit(self, docs: list[list[str]], labels: list[str], alpha: float = 1.0) -> None:
        """Learn the model from token lists and their class labels.

        log P(c)     = log(N_c / N)                                   N_c = training docs in class c
        log P(t | c) = log((T_ct + alpha) / (Σ_t' T_ct' + alpha·|V|))  T_ct = count of t in class c docs
        alpha = 1 is add-one (Laplace) smoothing; |V| = training vocabulary size.
        """
        self.alpha = alpha
        self.classes = sorted(set(labels))
        self.vocab = {t for doc in docs for t in doc}

        doc_counts = Counter(labels)
        term_counts = {c: Counter() for c in self.classes}  # T_ct for every class
        for doc, label in zip(docs, labels):
            term_counts[label].update(doc)

        for c in self.classes:
            self.log_prior[c] = math.log(doc_counts[c] / len(docs))
            denominator = sum(term_counts[c].values()) + alpha * len(self.vocab)
            # Every vocabulary word gets a probability, even if it never appeared in class c (count 0).
            self.log_likelihood[c] = {
                t: math.log((term_counts[c][t] + alpha) / denominator) for t in self.vocab
            }

    def predict(self, tokens: list[str]) -> tuple[str, dict[str, float]]:
        """Return (best class, {class: log score}).

        score(c) = log P(c) + Σ_{t in doc} log P(t | c)     (words not in the training vocabulary are skipped)
        prediction = argmax_c score(c)
        """
        known = [t for t in tokens if t in self.vocab]
        scores = {
            c: self.log_prior[c] + sum(self.log_likelihood[c][t] for t in known)
            for c in self.classes
        }
        best = max(scores, key=scores.get)
        return best, scores

    def predict_proba(self, tokens: list[str]) -> dict[str, float]:
        """Turn log scores into probabilities with a numerically stable softmax.

        P(c | doc) = exp(score(c) - m) / Σ_c' exp(score(c') - m),   m = max_c score(c)
        Subtracting m changes nothing mathematically but stops exp() from underflowing to 0
        (scores are often around -2000, and exp(-2000) is 0 on a computer).
        """
        _, scores = self.predict(tokens)
        m = max(scores.values())
        exps = {c: math.exp(s - m) for c, s in scores.items()}
        total = sum(exps.values())
        return {c: e / total for c, e in exps.items()}

    def top_words(self, c: str, n: int = 15) -> list[tuple[str, float]]:
        """Words that most strongly indicate class c.

        ratio(t) = log P(t | c) - max_{c' != c} log P(t | c')
        i.e. how many times (in log space) more likely t is in c than in the best other class.
        """
        others = [o for o in self.classes if o != c]
        ratios = {
            t: self.log_likelihood[c][t] - max(self.log_likelihood[o][t] for o in others)
            for t in self.vocab
        }
        return sorted(ratios.items(), key=lambda item: (-item[1], item[0]))[:n]

    def explain(self, tokens: list[str], c: str, n: int = 10) -> list[tuple[str, float]]:
        """Words in this document that most favour class c over the runner-up class r.

        r = the class other than c with the highest score for this document
        contribution(t) = count(t in doc) · (log P(t | c) - log P(t | r))
        Only words with a positive contribution (pushing towards c) are returned.
        """
        _, scores = self.predict(tokens)
        runner_up = max((o for o in self.classes if o != c), key=scores.get)
        counts = Counter(t for t in tokens if t in self.vocab)
        contributions = {
            t: k * (self.log_likelihood[c][t] - self.log_likelihood[runner_up][t])
            for t, k in counts.items()
        }
        positive = [(t, v) for t, v in contributions.items() if v > 0]
        return sorted(positive, key=lambda item: (-item[1], item[0]))[:n]

    def save(self, path: str = NB_PKL) -> None:
        """Pickle the model's attributes (a plain dict) to path."""
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str = NB_PKL) -> "NaiveBayes":
        """Create a NaiveBayes model from a file written by save()."""
        model = cls()
        with open(path, "rb") as f:
            model.__dict__.update(pickle.load(f))
        return model


if __name__ == "__main__":
    articles = load_articles()
    tokens = load_tokens()
    train = articles[articles["split"] == "train"]
    test = articles[articles["split"] == "test"]

    # Train on the train split ONLY; the test articles stay unseen until evaluation.
    model = NaiveBayes()
    model.fit([tokens[d] for d in train["doc_id"]], list(train["category"]), alpha=1.0)
    model.save(NB_PKL)

    predictions = [model.predict(tokens[d])[0] for d in test["doc_id"]]
    truth = list(test["category"])
    print(f"Trained on {len(train)} articles, vocabulary {len(model.vocab)} words; saved {NB_PKL}")
    print(f"Test accuracy: {accuracy_score(truth, predictions):.4f} on {len(test)} articles\n")
    print(classification_report(truth, predictions, digits=4))
    print("Confusion matrix (rows = true, columns = predicted):")
    print("classes:", model.classes)
    print(confusion_matrix(truth, predictions, labels=model.classes))

    print("\nTop 15 words per class (log-likelihood ratio vs best other class):")
    for c in model.classes:
        words = ", ".join(f"{t} ({r:.2f})" for t, r in model.top_words(c, 15))
        print(f"{c}: {words}")
