# Validation against library implementations

## 1. BM25: my `bm25_search` vs `rank_bm25.BM25Okapi` (k1=1.5, b=0.75, top 10)

| Query | Overlap /10 | Top result matches | My top doc | BM25Okapi top doc | Max score diff with Okapi idf |
|---|---|---|---|---|---|
| interest rates bank | 10 | yes | 136 | 136 | 3.6e-15 |
| chelsea premiership | 10 | yes | 1470 | 1470 | 1.8e-15 |
| mobile phone games | 10 | yes | 1788 | 1788 | 3.6e-15 |
| oil prices economy | 10 | yes | 29 | 29 | 1.8e-15 |
| oscar nominations best actress | 10 | yes | 559 | 559 | 3.6e-15 |
| film box office | 10 | yes | 550 | 550 | 3.6e-15 |
| general election labour tory | 10 | yes | 1177 | 1177 | 3.6e-15 |
| immigration asylum policy | 10 | yes | 1052 | 1052 | 3.6e-15 |
| world cup rugby | 10 | yes | 1610 | 1610 | 1.8e-15 |
| broadband internet users | 10 | yes | 1868 | 1868 | 1.8e-15 |
| **Average** | **10.0** | **10/10** | | | |

The term-frequency and length parts of the two formulas are identical; only idf differs:

- mine: `idf = ln((N - df + 0.5) / (df + 0.5) + 1)`, always positive;
- BM25Okapi: `idf = ln((N - df + 0.5) / (df + 0.5))`, negative when df > N/2, and negative values are
  replaced by `epsilon * average idf` (epsilon = 0.25).

The last column re-scores every document with my formula but BM25Okapi's idf values: the scores are
identical up to floating-point rounding, so idf is the only difference. Because of the `+ 1`, my idf
gives common words relatively more weight (compare `world` in the table below), which can swap a
document near rank 10.

| Term | df | My idf | Okapi raw idf | Okapi idf used |
|---|---|---|---|---|
| said | 1746 | 0.165 | -1.721 | 1.552 |
| world | 630 | 1.183 | 0.818 | 0.818 |
| bank | 178 | 2.445 | 2.355 | 2.355 |
| chelsea | 65 | 3.448 | 3.416 | 3.416 |
| asylum | 26 | 4.353 | 4.340 | 4.340 |

## 2. Naïve Bayes: my `NaiveBayes` vs scikit-learn `MultinomialNB(alpha=1.0)`

| Model | Test accuracy |
|---|---|
| My NaiveBayes | 96.60% |
| scikit-learn MultinomialNB | 96.60% |

Predictions agree on **412 / 412 test articles (100.00%)**.
