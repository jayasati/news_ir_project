# PROGRESS

One section per step. Each step ends by running its tests, updating its section here,
appending to reports/viva_notes.md and committing as "Step N: <name>".

## Step 1: Setup

- **Status:** Done
- **Files:** `data/`, `src/__init__.py`, `models/`, `tests/`, `reports/figures/`, `requirements.txt`, `.venv/`, `CLAUDE.md`, `PROGRESS.md`, `reports/viva_notes.md`, `.gitignore`
- **Key functions (with signatures):** None (setup only).
- **Results:** Virtual environment `.venv` (Python 3.13.5) with all requirements installed; NLTK English stopwords downloaded (198 words). BBC dataset in `data/bbc/`: business 510, entertainment 386, politics 417, sport 511, tech 401 — total 2,225 `.txt` files (matches the expected count).
- **Notes:** Dataset downloaded from http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip (linked from http://mlg.ucd.ie/datasets/bbc.html); the zip also contains `data/bbc/README.TXT`. `data/` is git-ignored except `data/queries.csv` and `data/qrels.csv`, so the articles are never committed. No tests in this step. Windows commands: `py -3.13 -m venv .venv`, `.venv\Scripts\python.exe -m pip install -r requirements.txt`, `.venv\Scripts\python.exe -c "import nltk; nltk.download('stopwords')"`.

## Step 2: Loader

- **Status:** Done
- **Files:** `src/loader.py`, `src/eda.py`, `tests/test_loader.py`; outputs `data/articles.csv`, `reports/figures/class_distribution.png`, `reports/figures/doc_length.png`
- **Key functions (with signatures):**
  - `read_text_file(path: Path) -> str`: UTF-8 with latin-1 fallback
  - `split_title_and_text(content: str) -> tuple[str, str]`: first non-empty line is the title, the rest (stripped) is the text
  - `load_raw(data_dir: str = "data/bbc") -> pd.DataFrame`: columns category, title, text; folders and files sorted so the order is the same on every run
  - `remove_duplicates(df: pd.DataFrame) -> pd.DataFrame`: drops, in order, exact copies (same title + text), same category + title, same category + text, and near-copies; keeps the first; prints counts per rule and category
  - `near_copy_rows(df: pd.DataFrame, threshold: float = 0.95) -> list`: index labels of articles whose lowercase word set has Jaccard ≥ 0.95 with an earlier article in the same category
  - `add_split(df: pd.DataFrame) -> pd.DataFrame`: adds split = train/test (stratified 80/20, random_state=42)
  - `load_articles() -> pd.DataFrame`: reads `data/articles.csv` (all later steps use this)
  - `src/eda.py`: `plot_class_distribution(df, path) -> None`, `plot_doc_length(df, path) -> None`
- **Results:**
  - Duplicates removed (exact + same title + same text + near-copy): business 510→498 (7+2+3+0), entertainment 386→362 (17+4+3+0), politics 417→388 (14+6+8+1), sport 511→488 (6+5+11+1), tech 401→322 (54+14+10+1); **total 2,225→2,058** (98 + 31 + 35 + 3 = 167).
  - Split: **train 1,646 / test 412**. Per category (train/test): business 398/100, entertainment 290/72, politics 310/78, sport 390/98, tech 258/64.
  - Words per article (whitespace count, title + text): min 89, median 328, mean 381, max 4,432.
  - `data/articles.csv` columns: doc_id (0..2057), category, title, text, split.
  - Tests: 7 in `tests/test_loader.py`, all pass.
- **Notes:** In this copy of the dataset, every file decoded as UTF-8, so the latin-1 fallback is kept only as a safety net. No duplicate appears in two different categories. The de-duplication grew in three rounds, each found by looking at real output and each followed by a full rebuild:
    (1) after Step 5, **same category + title** (31 articles: earlier/later versions of one story, 44–99.95% similar text, all checked by hand; 9 pairs split across train/test);
    (2) during Step 8, **same category + text** (35 articles: identical text with a re-worded headline, e.g. "Tory backing for ID cards" / "Conservative backing for ID cards"; 11 pairs split across train/test);
    (3) during Step 8, **near-copies** (3 articles: re-titled copies with a typo fixed or one word changed, e.g. "Child access laws shake-up" / "Child access law shake-up planned"; all 3 split across train/test). This rule uses word-set Jaccard ≥ 0.95 on lowercase whitespace-split words (a duplicate check only, not a retrieval tokenizer). The three removed pairs score 0.985–1.0; the next most similar pair scores 0.85 and is a genuinely updated story ("talks will take place" vs "talks have ended"), so it is kept. Pairs below that are different articles about the same event. Duplicates are removed *before* doc_id and the split are assigned, so ids have no gaps and no article can be in both train and test. The whitespace word count in `eda.py` is only for the length plot; it is not a tokenizer used for retrieval or classification. The tests read `data/articles.csv`, so run `python -m src.loader` before `pytest`.

## Step 3: Preprocessing

- **Status:** Done
- **Files:** `src/preprocess.py`, `tests/test_preprocess.py`; output `data/tokens.pkl`
- **Key functions (with signatures):**
  - `preprocess(text: str) -> list[str]`: lowercase → `[a-z]+` tokens → drop NLTK English stopwords and one-letter tokens → Porter stem. `STOPWORDS`, `STEMMER` and `TOKEN_PATTERN` are built once at module level; stopwords are downloaded if missing.
  - `tokenize_all() -> dict[int, list[str]]`: `preprocess(title + " " + text)` for every article, keyed by doc_id; saves `data/tokens.pkl`
  - `load_tokens() -> dict[int, list[str]]`: reads `data/tokens.pkl` (later steps use this)
- **Results:**
  - 2,058 articles tokenized; total 444,650 tokens.
  - **Vocabulary size: 18,955 distinct stems.**
  - **Average length: 216.1 tokens per article** (vs 381 raw whitespace words in Step 2).
  - Top 20 stems: said 6655, year 3094, mr 2804, would 2358, also 1964, us 1840, peopl 1798, new 1793, one 1774, time 1540, game 1526, say 1446, could 1370, use 1345, last 1305, make 1294, first 1243, two 1188, govern 1185, world 1153.
  - Tests: 4 in `tests/test_preprocess.py`, all pass.
- **Notes:** Stopwords are removed *before* stemming because the NLTK list holds whole words. "us" is not in the NLTK stopword list; here it is almost always the country "US", so keeping it is useful. Words like "said", "mr", "would" are very frequent but not stopwords; IDF (Step 4/5) will give them low weight. "U.K." becomes "u" + "k", which are dropped as one-letter tokens. `data/tokens.pkl` must be rebuilt if `preprocess()` ever changes.

## Step 4: Index

- **Status:** Done
- **Files:** `src/indexer.py`, `tests/conftest.py` (toy corpus fixture `toy_docs`), `tests/test_indexer.py`; output `models/index.pkl`
- **Key functions (with signatures):** class `InvertedIndex` with attributes `postings: dict[str, dict[int, int]]` (term → {doc_id: tf}), `doc_len: dict[int, int]`, `cf: dict[str, int]`, `doc_norm: dict[int, float]`, `N: int`, `total_tokens: int`, `avgdl: float`
  - `build(self, docs: dict[int, list[str]]) -> None`: fills everything, then calls `_compute_doc_norms()`
  - `_compute_doc_norms(self) -> None`: `doc_norm[d] = sqrt(Σ_t w(t,d)²)`, `w(t,d) = (1 + log10 tf) · log10(N/df)`
  - `df(self, term: str) -> int`
  - `candidates(self, q_terms: list[str]) -> set[int]`: union of the terms' postings
  - `boolean_and(self, query: str) -> set[int]`, `boolean_or(self, query: str) -> set[int]`, `boolean_not(self, query_a: str, query_b: str) -> set[int]` (= AND(a) − OR(b)); all run `preprocess()` on their input
  - `save(self, path: str = "models/index.pkl") -> None`, `InvertedIndex.load(path: str = "models/index.pkl") -> InvertedIndex` (classmethod)
- **Results:**
  - All 2,058 articles (train + test) indexed: N = 2,058, vocabulary 18,955 terms, 444,650 tokens, avgdl = 216.06.
  - df("elect") = 215; first 10 postings (doc_id, tf): (6, 2), (34, 1), (75, 1), (100, 1), (101, 2), (105, 6), (106, 2), (120, 1), (133, 1), (135, 1).
  - 292,454 postings entries in total, vs 2,058 × 18,955 ≈ 39.0 million cells in a full term-document matrix (only 0.75% non-zero). 8,081 terms occur in just one article.
  - Boolean examples: "election AND labour" → 137 docs; "election OR labour" → 294; "election NOT labour" → 78 (137 + 78 = 215 = df("elect")).
  - No document has doc_norm = 0. Tests: 7 in `tests/test_indexer.py`, all pass.
- **Notes:** `save()` pickles the attribute dict (`self.__dict__`), and `load()` copies it into a new object. The toy corpus in `conftest.py` uses doc_ids 1, 2, 3 for D1, D2, D3; its tokens are already processed, so the Boolean tests use words that Porter stemming leaves unchanged ("share", "market", "cricket", "launch", "new"). Note that "apple" → "appl" and "iphone" → "iphon", so a raw query "apple" would *not* match the toy tokens. The index is built on all articles because search should be able to find any article; only Naïve Bayes needs the train/test split.

## Step 5: Rankers

- **Status:** Done
- **Files:** `src/rankers.py`, `src/demo_search.py`, `tests/test_rankers.py`
- **Key functions (with signatures):**
  - `tfidf_search(index: InvertedIndex, q_terms: list[str], k: int = 10) -> list[tuple[int, float]]`: `Σ_t w(t,q)·w(t,d) / doc_norm[d]`, `w = (1 + log10 tf)·log10(N/df)`
  - `bm25_search(index, q_terms, k=10, k1=1.5, b=0.75) -> list[tuple[int, float]]`: `Σ_t idf·tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))`, `idf = ln((N − df + 0.5)/(df + 0.5) + 1)`
  - `lm_search(index, q_terms, k=10, smoothing="dirichlet", mu=2000, lam=0.1) -> list[tuple[int, float]]`: `Σ_t ln P(t|d)`; Dirichlet `(tf + mu·P(t|C))/(|d| + mu)`, JM `(1 − lam)·tf/|d| + lam·P(t|C)`, `P(t|C) = cf/total_tokens`
  - `top_k(scores: dict[int, float], k: int) -> list[tuple[int, float]]`: sort by score descending, ties broken by doc_id
  - `search(index: InvertedIndex, query: str, model: str, k: int = 10) -> list[tuple[int, float]]`: `preprocess(query)` then `RANKERS[model]`
  - `RANKERS = {"TF-IDF": tfidf_search, "BM25": bm25_search, "Unigram LM": lm_search}`
- **Results:**
  - Toy query ["apple", "iphone"]: TF-IDF D1 0.3061 / D2 0.0367; BM25 (k1=1.2, b=0.75) D1 1.4980 / D2 0.4853; LM JM (λ=0.1) D1 −2.8835 / D2 −6.2930. D3 is never returned. Tests: 6 in `tests/test_rankers.py`, all pass.
  - Demo output (`python -m src.demo_search`):

```
=== Query: "interest rates bank" ===
--- TF-IDF ---
1.    0.3643  [business] Bank set to leave rates on hold  (doc 136)
2.    0.3405  [business] Bank voted 8-1 for no rate change  (doc 150)
3.    0.2854  [business] Australia rates at four year high  (doc 100)
4.    0.2755  [business] House prices show slight increase  (doc 152)
5.    0.2541  [business] Economy 'strong' in election year  (doc 213)
--- BM25 ---
1.   13.5518  [business] Bank set to leave rates on hold  (doc 136)
2.   13.3311  [business] Bank voted 8-1 for no rate change  (doc 150)
3.   13.2379  [business] Bank holds interest rate at 4.75%  (doc 287)
4.   12.6205  [business] Bank opts to leave rates on hold  (doc 118)
5.   12.6152  [business] Australia rates at four year high  (doc 100)
--- Unigram LM ---
1.  -16.4852  [business] Bank holds interest rate at 4.75%  (doc 287)
2.  -16.8815  [business] US interest rates increased to 2%  (doc 120)
3.  -16.9314  [business] Bank opts to leave rates on hold  (doc 118)
4.  -16.9816  [business] Bank voted 8-1 for no rate change  (doc 150)
5.  -17.1949  [business] Bank set to leave rates on hold  (doc 136)

=== Query: "chelsea premiership" ===
--- TF-IDF ---
1.    0.4504  [sport] Newcastle line up Babayaro  (doc 1518)
2.    0.3846  [sport] Beckham relief as Real go through  (doc 1480)
3.    0.3801  [sport] Ferguson rues failure to cut gap  (doc 1493)
4.    0.3785  [sport] Gallas sees two-horse race  (doc 1451)
5.    0.3779  [sport] Cudicini misses Carling Cup final  (doc 1417)
--- BM25 ---
1.   12.4168  [sport] Houllier praises Benitez regime  (doc 1470)
2.   11.7196  [sport] Cup holders Man Utd visit Everton  (doc 1426)
3.   11.3313  [sport] Newcastle line up Babayaro  (doc 1518)
4.   11.0670  [sport] Ferguson rues failure to cut gap  (doc 1493)
5.   11.0070  [sport] Thompson says Gerrard should stay  (doc 1420)
--- Unigram LM ---
1.  -12.3458  [sport] Houllier praises Benitez regime  (doc 1470)
2.  -12.5485  [sport] Chelsea denied by James heroics  (doc 1382)
3.  -12.8727  [sport] Chelsea hold Arsenal  (doc 1487)
4.  -12.9043  [sport] Desailly backs Blues revenge trip  (doc 1424)
5.  -13.0911  [sport] Cup holders Man Utd visit Everton  (doc 1426)

=== Query: "mobile phone games" ===
--- TF-IDF ---
1.    0.4096  [tech] Home phones face unclear future  (doc 1841)
2.    0.3724  [tech] Half of UK's mobiles 'go online'  (doc 1979)
3.    0.3656  [tech] Mobile gig aims to rock 3G  (doc 1903)
4.    0.3532  [business] Brussels raps mobile call charges  (doc 112)
5.    0.3495  [tech] Mobile gaming takes off in India  (doc 1912)
--- BM25 ---
1.   14.7046  [tech] Moving mobile improves golf swing  (doc 1788)
2.   14.3736  [tech] Mobile audio enters new dimension  (doc 2012)
3.   14.2610  [tech] New consoles promise big problems  (doc 2053)
4.   14.2138  [tech] Cheaper chip for mobiles  (doc 1784)
5.   14.0494  [tech] Mobile gaming takes off in India  (doc 1912)
--- Unigram LM ---
1.  -14.8255  [tech] New consoles promise big problems  (doc 2053)
2.  -15.3198  [tech] Mobile games come of age  (doc 1827)
3.  -15.3449  [tech] 'Friends fear' with lost mobiles  (doc 1801)
4.  -15.5014  [tech] Finding new homes for old phones  (doc 1752)
5.  -15.6112  [tech] Mobile music challenges 'iPod age'  (doc 2046)
```
- **Notes:**
  - TF-IDF divides by the document norm only, not by the query norm. The query norm is the same for every document of one query, so the ranking is identical to full cosine. Full cosine on the toy example would give D1 0.60 / D2 0.07.
  - Only candidates (docs containing at least one query term) are scored. LM query terms with cf = 0 are skipped. Repeated query terms count more than once in BM25 and LM; TF-IDF uses `1 + log10 qtf`.
  - The toy BM25 test uses k1=1.2; the project default is k1=1.5.
  - **Near-duplicates (resolved):** the first demo run showed the same title twice ("Desailly backs Blues revenge trip", "Mobile games come of age"): near-copies that the exact-match rule in Step 2 missed (31 pairs, 9 split across train/test). Step 2 now also drops same-category + same-title, same-category + same-text and near-copy repeats; the demo above is the run on the final 2,058 articles (no repeated stories).

## Step 6: Naïve Bayes

- **Status:** Done
- **Files:** `src/classifier.py`, `tests/test_classifier.py`; output `models/nb.pkl`
- **Key functions (with signatures):** class `NaiveBayes` with attributes `classes: list[str]`, `vocab: set[str]`, `log_prior: dict[str, float]`, `log_likelihood: dict[str, dict[str, float]]`, `alpha: float`
  - `fit(self, docs: list[list[str]], labels: list[str], alpha: float = 1.0) -> None`: `log P(c) = log(N_c/N)`, `log P(t|c) = log((T_ct + α)/(Σ_t' T_ct' + α|V|))`
  - `predict(self, tokens: list[str]) -> tuple[str, dict[str, float]]`: `argmax_c log P(c) + Σ_t log P(t|c)`, skipping words not in the training vocabulary
  - `predict_proba(self, tokens: list[str]) -> dict[str, float]`: softmax `exp(s_c − m)/Σ exp(s_c' − m)`, m = max score
  - `top_words(self, c: str, n: int = 15) -> list[tuple[str, float]]`: `log P(t|c) − max_{c'≠c} log P(t|c')`
  - `explain(self, tokens: list[str], c: str, n: int = 10) -> list[tuple[str, float]]`: `count(t)·(log P(t|c) − log P(t|r))`, r = runner-up class for this document; positive contributions only
  - `save(self, path: str = "models/nb.pkl") -> None`, `NaiveBayes.load(path: str = "models/nb.pkl") -> NaiveBayes` (classmethod)
- **Results:**
  - Trained on the 1,646 train articles only (vocabulary 17,287 stems), tested on the 412 test articles.
  - **Test accuracy: 96.60%** (398/412). Macro F1 0.9643.
  - Per class (precision / recall / F1): business 0.989 / 0.940 / 0.964; entertainment 1.000 / 0.917 / 0.957; politics 0.917 / 0.987 / 0.951; sport 1.000 / 0.990 / 0.995; tech 0.914 / 1.000 / 0.955.
  - Confusion matrix (rows true, columns predicted; order business, entertainment, politics, sport, tech): [94 0 3 0 3], [0 66 4 0 2], [0 0 77 0 1], [1 0 0 97 0], [0 0 0 0 64].
  - Top 15 words per class:
    - business: yuko, gm, oil, glazer, sharehold, lse, fiat, rosneft, boers, yugansk, gazprom, ebber, boe, worldcom, khodorkovski
    - entertainment: actress, oscar, singer, foxx, actor, comedi, eastwood, rapper, staunton, scorses, ballet, swank, drake, sideway, album
    - politics: tori, lib, blunkett, asylum, ukip, dem, kilroy, blair, prescott, conserv, kennedi, liber, detent, detaine, goldsmith
    - sport: roddick, cup, refere, rugbi, nadal, mourinho, seed, olymp, chelsea, kenteri, athen, championship, henman, arsen, iaaf
    - tech: blog, spam, nintendo, gamer, gadget, user, virus, portabl, microsoft, wireless, laptop, browser, psp, comput, devic
  - IIR Example 13.1 reproduced: P(Chinese|c) = 3/7, P(Tokyo|j) = 2/9, test doc → c. Tests: 7 in `tests/test_classifier.py`, all pass.
  - Before the last two de-duplication rounds (2,096 articles) accuracy was 97.14% (408/420). Removing the 38 re-titled and near copies (14 pairs had one copy on each side of the split) lowered it to 96.60%, so the higher number was partly leakage.
- **Notes:** Accuracy is inside the expected 95–98% band, so there is no sign of leakage (near-duplicates were removed in Step 2) or of a preprocessing problem. The 14 errors are mostly genuinely mixed articles: "Ferdinand casts doubt over Glazer" (sport → business: glazer, club, stake), "News Corp eyes video games market" (business → tech: game, video), "CSI shows give 'unrealistic view'" (entertainment → tech: lab, scienc, test), "Call to overhaul UK state pension" (business → politics), "Row threatens Hendrix museum plan" (entertainment → politics: mayor). Entertainment has the lowest recall (0.917): arts stories that read like politics or tech. 13 of the 14 errors still get probability > 0.99: Naïve Bayes probabilities are over-confident because the independence assumption counts every correlated word as separate evidence. Only training vocabulary words are used at prediction time, so unseen test words are skipped.

## Step 7: Validation

- **Status:** Done
- **Files:** `src/validate.py`; output `reports/validation.md`
- **Key functions (with signatures):**
  - `QUERIES`: the 3 demo queries + 7 more (oil prices economy; oscar nominations best actress; film box office; general election labour tory; immigration asylum policy; world cup rugby; broadband internet users), covering all five categories
  - `okapi_top10(okapi: BM25Okapi, q_terms: list[str]) -> list[int]`: BM25Okapi's top 10 doc_ids (stable sort, ties → lower doc_id)
  - `bm25_with_okapi_idf(index: InvertedIndex, okapi: BM25Okapi, q_terms: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray`: my BM25 formula with Okapi's idf values, for every doc
  - `validate_bm25(index: InvertedIndex, tokens: dict[int, list[str]]) -> tuple[list[dict], list[dict]]`: overlap@10, top-1 match, max score difference, idf comparison
  - `validate_nb(articles, tokens: dict[int, list[str]]) -> dict`: my `NaiveBayes` (models/nb.pkl) vs `MultinomialNB(alpha=1.0)` + `CountVectorizer(analyzer=lambda x: x)`
  - `to_markdown(bm25_rows: list[dict], idf_rows: list[dict], nb: dict) -> str`
- **Results:**
  - **BM25: average overlap 10.0 / 10, top result matches on 10 / 10 queries** (every query overlaps 10/10).
  - Re-scoring with Okapi's idf gives a max score difference ≤ 3.6e-15 over all 2,058 docs for every query, so the tf, k1, b and length-normalisation parts are identical and idf is the only difference.
  - idf comparison (mine / Okapi raw / Okapi used): said (df 1746) 0.165 / −1.721 / 1.552; world (630) 1.183 / 0.818 / 0.818; bank (178) 2.445 / 2.355 / 2.355; chelsea (65) 3.448 / 3.416 / 3.416; asylum (26) 4.353 / 4.340 / 4.340.
  - **Naïve Bayes: my accuracy 96.60%, scikit-learn MultinomialNB 96.60%, predictions agree on 412 / 412 test articles (100.00%).**
  - Both thresholds (NB agreement ≥ 99%, BM25 overlap ≥ 8/10) pass, so no fixes were needed.
  - On the earlier 2,096-article data the BM25 overlap was 9.7 / 10 (three queries at 9/10, one document swapped near rank 10), with the same ≤ 3.6e-15 idf proof.
- **Notes:** The idf difference is the "+ 1" inside my log (always positive) vs Okapi's raw log, which is negative for df > N/2 and is then floored to 0.25 × average idf. None of the 10 queries has a term with df > N/2, so the floor never applies to them. The "+ 1" gives common words relatively more weight ("world", df 630: 1.183 mine vs 0.818 Okapi), which can swap a document near rank 10 (it did for three queries on the 2,096-article data). The floor matters for very common words: "said" (in 85% of articles) gets idf 1.552 in Okapi but 0.165 in mine. The Windows console may show "Naïve" as "Na�ve" when printing; `reports/validation.md` itself is UTF-8 and correct.

## Step 8: Query pool

- **Status:** Done (pool built on the final 2,058 articles); waiting for manual relevance judgements in `data/pool_to_judge.csv`
- **Files:** `data/queries.csv` (committed), `src/pool.py`, `tests/test_pool.py`; output `data/pool_to_judge.csv` (git-ignored, contains article text)
- **Key functions (with signatures):**
  - `load_queries() -> pd.DataFrame`: reads `data/queries.csv` (query_id, query, intended_category)
  - `pool_for_query(index: InvertedIndex, query: str, k: int = 10) -> list[int]`: union of the top 10 of TF-IDF, BM25 and Unigram LM, sorted by doc_id
  - `rarest_term(index: InvertedIndex, q_terms: list[str]) -> str | None`: query term with the highest idf (lowest df > 0)
  - `suggest_rel(doc_tokens: set[str], category: str, intended_category: str, rare_term: str | None) -> int`: 1 if same category AND contains the rarest term, else 0
  - `build_pool() -> pd.DataFrame`: columns query_id, query, doc_id, title, category, snippet (first 300 chars, line breaks → spaces), suggested_rel, rel (empty)
- **Results:**
  - 20 queries, 4 per category. business: interest rates bank; oil prices rise; stock market shares; airline job cuts. entertainment: oscar nominations; box office film; album chart singer; television comedy show. politics: general election labour; asylum immigration policy; tory tax plans; house lords reform. sport: chelsea premiership; six nations rugby; australian open tennis; olympic athletics drugs. tech: mobile phone games; broadband internet users; computer virus security; apple ipod sales.
  - Pool size per query (query_id: size): 1: 13, 2: 15, 3: 15, 4: 17, 5: 15, 6: 11, 7: 12, 8: 14, 9: 16, 10: 12, 11: 13, 12: 13, 13: 16, 14: 16, 15: 14, 16: 16, 17: 18, 18: 17, 19: 14, 20: 12.
  - **Total rows to judge: 289** (min 11, max 18, mean 14.4 per query; the maximum possible is 30, so the three rankers agree a lot).
  - 283 / 289 pooled docs are in the query's intended category; suggested_rel = 1 for 279 / 289.
  - Tests: 3 in `tests/test_pool.py`; all 34 project tests pass.
- **Notes:**
  - The pool is sorted by doc_id, so the judge cannot see which ranker found a document or at what rank.
  - suggested_rel is weak evidence: it is 1 for 97% of rows, so it cannot replace reading the snippet. Judge on whether the article is *about* the query topic.
  - `pool_to_judge.csv` is written as UTF-8 with a byte-order mark so Excel shows £ etc. correctly; when saving from Excel use "CSV UTF-8". pandas reads it normally.
  - **LLM draft judgements (added after the step was committed):** at the student's request, Claude (claude-opus-5-5, in the Claude Code session) read every pooled article in full and added two columns to `data/pool_to_judge.csv`: `llm_rel` (0/1/2) and `llm_reason` (one line). Grading rule: 2 = mainly about the query topic, 1 = clearly relevant but only part of the story, 0 = not relevant or only mentioned in passing. Rows were read in doc_id order without seeing which ranker found them; `suggested_rel` was not shown. Result: 122 × 2, 95 × 1, 72 × 0. 63 of the 279 rows with suggested_rel = 1 were graded 0, which confirms suggested_rel is a weak hint. Every query has at least one relevant document; the fewest highly relevant are "airline job cuts" and "apple ipod sales" (1 each). In Step 9 the student stated they agreed with all `llm_rel` grades, so `rel` = `llm_rel` for all 289 rows (0 changed). **Do not rerun `python -m src.pool`**: it rebuilds the file and would erase `llm_rel`, `llm_reason` and `rel`.
  - **Resolved before judging:** the first pool (284 rows) showed "Gadget show heralds MP3 season" and "... MP3 Christmas" side by side: identical text with a re-worded headline. Step 2 now also removes same-category + same-text copies (35) and near-copies (3), and Steps 2–8 were rebuilt, so doc_ids changed. **Only the current `pool_to_judge.csv` (289 rows, 2,058 articles) is valid for judging.**

## Step 9: Evaluation

- **Status:** Done
- **Files:** `src/evaluate.py`, `tests/test_evaluate.py`; outputs `data/qrels.csv` (committed), `reports/results.md`, `reports/figures/bm25_sweep.png`, `reports/figures/lm_mu_sweep.png`, `reports/figures/confusion_matrix.png`
- **Key functions (with signatures):**
  - `build_qrels() -> pd.DataFrame` (writes `data/qrels.csv`; refuses to run if any `rel` is empty), `load_qrels() -> dict[int, dict[int, int]]`
  - `precision_at_k(ranked: list[int], qrels: dict[int, int], k: int) -> float`, `recall_at_k(ranked, qrels, k) -> float`
  - `average_precision(ranked: list[int], qrels: dict[int, int]) -> float`: `(1/R) Σ P@i` over relevant ranks; unretrieved relevant docs add 0
  - `mean_average_precision(runs: dict[int, list[int]], qrels: dict[int, dict[int, int]]) -> float`
  - `dcg_at_k(gains: list[int], k: int) -> float` (`Σ rel_i / log2(i+1)`), `ndcg_at_k(ranked, qrels, k) -> float`
  - `sign_test_p(wins: int, losses: int) -> float`: two-sided sign test with `math.comb`
  - `run_queries(ranker, index, queries, **params) -> dict[int, list[int]]` (depth 1000), `score_runs(runs, qrels) -> dict[str, float]`, `compare_per_query(runs_a, runs_b, qrels) -> tuple[int, int, int, float]`
  - `sweep_bm25(index, queries, qrels) -> pd.DataFrame`, `sweep_lm(index, queries, qrels) -> pd.DataFrame`, `evaluate_classifier(articles, tokens) -> dict`, plotting and report helpers
- **Results:**
  - Judgements: 289 pairs, 20 queries (0: 72, 1: 95, 2: 122); LLM-drafted, accepted unchanged by the student.
  - Retrieval (default parameters), P@5 / P@10 / R@10 / MAP / nDCG@10:
    - TF-IDF 0.870 / 0.790 / 0.743 / 0.813 / 0.777
    - **BM25 (k1 1.5, b 0.75) 0.960 / 0.850 / 0.815 / 0.929 / 0.857**
    - LM Dirichlet (mu 2000) 0.910 / 0.825 / 0.772 / 0.873 / 0.838
  - Per-query AP, sign test: BM25 vs TF-IDF 14 wins / 1 tie / 5 losses, p = 0.064; LM vs TF-IDF 11/1/8, p = 0.648; BM25 vs LM 15/1/4, p = 0.019 (not significant after Bonferroni for 3 tests, threshold 0.017). Largest gap: "mobile phone games" (TF-IDF 0.230, BM25 1.000, LM 0.603).
  - BM25 sweep MAP 0.908–0.936; best k1 = 2.0, b = 0.75 (0.936) vs default 0.929; b = 0.75 best for every k1; MAP rises with k1.
  - LM sweep MAP falls steadily as mu grows: mu 500 0.914, 1000 0.896, 2000 0.873, 3000 0.850.
  - Classification (test, 412 articles): accuracy 0.9660, macro-F1 0.9643; top confusions entertainment→politics 4, business→politics 3, business→tech 3, entertainment→tech 2, politics→tech 1.
  - Tests: 6 new (hand-worked P@k, R@k, AP = 5/9, MAP = 7/9, nDCG@5 ≈ 0.7985, sign test); all 40 pass.
- **Notes:** Documents outside the pool count as not relevant, so recall and MAP are optimistic and the parameter sweep is biased towards the default settings the pool came from; the sweep is tuned and scored on the same 20 queries. P@k and nDCG@10 are the most trustworthy numbers. The interpretation in `results.md` is generated from the numbers (with a Bonferroni check) so it cannot claim more than the tables show. Why TF-IDF fails on "mobile phone games": "game" is in 420 articles (mostly sport), so its idf is low; cosine favours short articles dense in "mobile"/"phone" (top 4 all about phones in general, rel 0), while BM25's tf saturation lets articles containing all three words win.

## Step 10: App

- **Status:** Done
- **Files:** `app.py` (project root), `tests/test_app.py`
- **Run the app:** `.venv\Scripts\python.exe -m streamlit run app.py` (or `streamlit run app.py` with the venv activated), from the project root; it opens at http://localhost:8501.
- **Key functions (with signatures):**
  - `load_resources() -> tuple[InvertedIndex, NaiveBayes, pd.DataFrame]`: `@st.cache_resource`; loads index, model and articles once and adds a `predicted` column (Naive Bayes on `load_tokens()`)
  - `badge(category: str) -> str`: coloured HTML label
  - `snippet_with_highlights(text: str, query_stems: set[str]) -> str`: 250 characters around the first query word; a word is highlighted when `preprocess(word)` matches a query stem
  - `search_tab(index, articles) -> None`: query, model (BM25 default / TF-IDF / Unigram LM), 5–50 results, filter on predicted category; ranks to depth 1000, filters, then keeps the top n
  - `classify_tab(model) -> None`: predicted category + runner-up, `predict_proba()` bar chart, top 10 words from `explain()`
  - `evaluation_tab() -> None`: renders `reports/results.md` and the three charts
- **Results:** Tests: 4 new (app starts without exception; search "interest rates bank" shows results; stopword-only query warns; classify tab predicts sport for a football sentence); all 44 pass. The real server also started and passed its health check.
- **Notes:** Every query and pasted text goes through `preprocess()` (also used for snippet highlighting, so "Elections" is highlighted for the query "election"). Predicted categories for *training* articles are in-sample (the model saw them), so mismatches are rarer there: 7 of 1,646 train vs 14 of 412 test. The category filter is applied after ranking so it never hides better-ranked articles. Messages are shown for an empty query, a query with no searchable words, no results, and pasted text with no known words. Unigram LM scores are negative log-probabilities; the app says so.

## Step 11: Report

- **Status:** Not started
- **Files:**
- **Key functions (with signatures):**
- **Results:**
- **Notes:**
