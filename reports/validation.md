# Validation against library implementations

## 1. BM25: my `bm25_search` vs `rank_bm25.BM25Okapi` (k1=1.5, b=0.75, top 10)

| Query | Overlap /10 | Top result matches | My top doc | BM25Okapi top doc | Max score diff with Okapi idf |
|---|---|---|---|---|---|
| interest rates bank | 10 | yes | 136 | 136 | 1.8e-15 |
| chelsea premiership | 10 | yes | 1487 | 1487 | 1.8e-15 |
| mobile phone games | 10 | yes | 1815 | 1815 | 3.6e-15 |
| oil prices economy | 9 | yes | 29 | 29 | 3.6e-15 |
| oscar nominations best actress | 9 | yes | 562 | 562 | 3.6e-15 |
| film box office | 10 | yes | 553 | 553 | 1.8e-15 |
| general election labour tory | 10 | yes | 1188 | 1188 | 3.6e-15 |
| immigration asylum policy | 10 | yes | 1058 | 1058 | 3.6e-15 |
| world cup rugby | 9 | yes | 1629 | 1629 | 1.8e-15 |
| broadband internet users | 10 | yes | 1896 | 1896 | 1.8e-15 |
| **Average** | **9.7** | **10/10** | | | |

The term-frequency and length parts of the two formulas are identical; only idf differs:

- mine: `idf = ln((N - df + 0.5) / (df + 0.5) + 1)`, always positive;
- BM25Okapi: `idf = ln((N - df + 0.5) / (df + 0.5))`, negative when df > N/2, and negative values are
  replaced by `epsilon * average idf` (epsilon = 0.25).

The last column re-scores every document with my formula but BM25Okapi's idf values: the scores are
identical up to floating-point rounding, so idf is the only difference. Because of the `+ 1`, my idf
gives common words relatively more weight (e.g. `world`, df 639: 1.188 vs 0.824), which is why a
document near rank 10 sometimes swaps.

| Term | df | My idf | Okapi raw idf | Okapi idf used |
|---|---|---|---|---|
| said | 1781 | 0.163 | -1.731 | 1.554 |
| bank | 179 | 2.458 | 2.369 | 2.369 |
| chelsea | 66 | 3.451 | 3.419 | 3.419 |
| asylum | 27 | 4.334 | 4.321 | 4.321 |

## 2. Naïve Bayes: my `NaiveBayes` vs scikit-learn `MultinomialNB(alpha=1.0)`

| Model | Test accuracy |
|---|---|
| My NaiveBayes | 97.14% |
| scikit-learn MultinomialNB | 97.14% |

Predictions agree on **420 / 420 test articles (100.00%)**.
