# Results

Relevance judgements (`data/qrels.csv`): 289 judged (query, document) pairs for 20 queries, graded 0/1/2 by an LLM (Claude) that read every pooled article, then reviewed by the student, who accepted all 289 drafts unchanged. Grades: 0: 72, 1: 95, 2: 122.
Relevant = rel >= 1 for P@k, recall and AP; nDCG uses the graded rel (0/1/2). Each ranker returns up to 1000 documents per query. Documents outside the judged pool count as not relevant.

## 1. Retrieval (default parameters)

TF-IDF cosine; BM25 k1=1.5, b=0.75; unigram LM with Dirichlet mu=2000. Means over the 20 queries.

| model | P@5 | P@10 | R@10 | MAP | nDCG@10 |
|---|---|---|---|---|---|
| TF-IDF | 0.870 | 0.790 | 0.743 | 0.813 | 0.777 |
| BM25 | 0.960 | 0.850 | 0.815 | 0.929 | 0.857 |
| LM (Dirichlet) | 0.910 | 0.825 | 0.772 | 0.873 | 0.838 |

Per-query AP (one row per query):

| query_id | query | TF-IDF | BM25 | LM (Dirichlet) |
|---|---|---|---|---|
| 1 | interest rates bank | 0.835 | 1.000 | 0.986 |
| 2 | oil prices rise | 0.885 | 0.992 | 0.863 |
| 3 | stock market shares | 0.786 | 0.936 | 0.843 |
| 4 | airline job cuts | 0.445 | 0.673 | 0.329 |
| 5 | oscar nominations | 0.778 | 0.829 | 0.900 |
| 6 | box office film | 0.957 | 1.000 | 0.992 |
| 7 | album chart singer | 0.988 | 0.983 | 0.975 |
| 8 | television comedy show | 0.961 | 0.941 | 0.898 |
| 9 | general election labour | 0.715 | 0.917 | 0.796 |
| 10 | asylum immigration policy | 1.000 | 0.988 | 0.975 |
| 11 | tory tax plans | 0.885 | 0.969 | 0.994 |
| 12 | house lords reform | 1.000 | 1.000 | 1.000 |
| 13 | chelsea premiership | 0.678 | 0.769 | 0.816 |
| 14 | six nations rugby | 0.641 | 0.859 | 0.888 |
| 15 | australian open tennis | 0.894 | 0.962 | 0.930 |
| 16 | olympic athletics drugs | 0.787 | 0.956 | 0.944 |
| 17 | mobile phone games | 0.230 | 1.000 | 0.603 |
| 18 | broadband internet users | 0.865 | 0.859 | 0.851 |
| 19 | computer virus security | 0.960 | 0.954 | 0.913 |
| 20 | apple ipod sales | 0.976 | 1.000 | 0.958 |

Head-to-head on per-query AP (wins / ties / losses for the first system; two-sided sign test, ties dropped):

| Comparison | Wins | Ties | Losses | Sign-test p |
|---|---|---|---|---|
| BM25 vs TF-IDF | 14 | 1 | 5 | 0.064 |
| LM (Dirichlet) vs TF-IDF | 11 | 1 | 8 | 0.648 |
| BM25 vs LM (Dirichlet) | 15 | 1 | 4 | 0.019 |

## 2. Parameter sweeps (MAP)

BM25 (rows k1, columns b):

| k1 | b=0.5 | b=0.75 | b=1.0 |
|---|---|---|---|
| 0.900 | 0.909 | 0.915 | 0.908 |
| 1.200 | 0.918 | 0.923 | 0.914 |
| 1.500 | 0.926 | 0.929 | 0.918 |
| 2.000 | 0.929 | 0.936 | 0.922 |

Best: k1=2.0, b=0.75 with MAP 0.936 (default k1=1.5, b=0.75: 0.929). Chart: `reports/figures/bm25_sweep.png`.

Unigram LM, Dirichlet mu:

| mu | MAP |
|---|---|
| 500 | 0.914 |
| 1000 | 0.896 |
| 2000 | 0.873 |
| 3000 | 0.850 |

Best: mu=500 with MAP 0.914 (default mu=2000: 0.873). Chart: `reports/figures/lm_mu_sweep.png`.

**Pool bias.** The pool was built from the top 10 of the three rankers at their *default* settings. A setting that brings up different documents gets them counted as not relevant even if a person would judge them relevant, so the sweep is biased towards the defaults and small MAP differences should not be over-read. The sweep was also tuned on the same 20 queries it is scored on (no held-out queries).

## 3. Classification (Naive Bayes, test split)

Test articles: 412. **Accuracy 0.9660**, **macro-F1 0.9643** (14 errors).

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| business | 0.989 | 0.940 | 0.964 | 100 |
| entertainment | 1.000 | 0.917 | 0.957 | 72 |
| politics | 0.917 | 0.987 | 0.951 | 78 |
| sport | 1.000 | 0.990 | 0.995 | 98 |
| tech | 0.914 | 1.000 | 0.955 | 64 |

Confusion matrix (rows true, columns predicted). Heatmap: `reports/figures/confusion_matrix.png`.

| true \ predicted | business | entertainment | politics | sport | tech |
|---|---|---|---|---|---|
| business | 94 | 0 | 3 | 0 | 3 |
| entertainment | 0 | 66 | 4 | 0 | 2 |
| politics | 0 | 0 | 77 | 0 | 1 |
| sport | 1 | 0 | 0 | 97 | 0 |
| tech | 0 | 0 | 0 | 0 | 64 |

Most common confusions (an example title for each):

| true | predicted | count | example title |
|---|---|---|---|
| entertainment | politics | 4 | Media seek Jackson 'juror' notes |
| business | politics | 3 | Golden rule 'intact' says ex-aide |
| business | tech | 3 | Card fraudsters 'targeting web' |
| entertainment | tech | 2 | CSI shows give 'unrealistic view' |
| politics | tech | 1 | UK firms 'embracing e-commerce' |

## 4. Interpretation

- **Retrieval.** Highest MAP: BM25 (0.929); TF-IDF 0.813, LM (Dirichlet) 0.873.
- BM25 vs TF-IDF: better on 14 queries, equal on 1, worse on 5; sign-test p = 0.064, so this is not a statistically reliable difference on 20 queries (p >= 0.05).
- LM (Dirichlet) vs TF-IDF: better on 11 queries, equal on 1, worse on 8; sign-test p = 0.648, so this is not a statistically reliable difference on 20 queries (p >= 0.05).
- BM25 vs LM (Dirichlet): better on 15 queries, equal on 1, worse on 4; sign-test p = 0.019, so this is suggestive (p < 0.05) but not significant once corrected for 3 tests (Bonferroni threshold 0.017).
- Largest per-query gap: "mobile phone games" (TF-IDF AP 0.230, BM25 AP 1.000, LM (Dirichlet) AP 0.603). One query like this moves a 20-query MAP by several points.
- **BM25 sweep.** MAP ranges from 0.908 to 0.936 (spread 0.028) over the 12 settings. The defaults are already among the best settings.
- **LM sweep.** MAP ranges from 0.850 to 0.914 over mu = 500-3000 and falls steadily as mu grows. Smaller mu means less smoothing towards the collection model, i.e. more weight on the words actually in each article. Pool bias favours the default mu = 2000, so a smaller mu winning anyway is a reasonably strong hint, though it is still tuned and scored on the same 20 queries.
- **Classification.** Naive Bayes reaches 96.6% accuracy and 0.964 macro-F1 on held-out articles; the errors are mostly articles that genuinely mix two topics (see the examples).
- **Caveats.** 20 queries is a small sample; relevance was judged on a depth-10 pool from these same rankers; the judgements were LLM-drafted and accepted unchanged by the student. Precision-oriented measures (P@5, P@10, nDCG@10) are the most trustworthy here because every top-10 document of the default rankers was judged; recall and MAP are optimistic because relevant articles outside the pool are unknown.
