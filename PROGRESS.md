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
  - `remove_duplicates(df: pd.DataFrame) -> pd.DataFrame`: drops exact copies (same title + text), then near-copies (same category + title); keeps the first; prints counts per category
  - `add_split(df: pd.DataFrame) -> pd.DataFrame`: adds split = train/test (stratified 80/20, random_state=42)
  - `load_articles() -> pd.DataFrame`: reads `data/articles.csv` (all later steps use this)
  - `src/eda.py`: `plot_class_distribution(df, path) -> None`, `plot_doc_length(df, path) -> None`
- **Results:**
  - Duplicates removed (exact + same-title): business 510→501 (7+2), entertainment 386→365 (17+4), politics 417→397 (14+6), sport 511→500 (6+5), tech 401→333 (54+14); total 2,225→2,096 (98 exact + 31 same-title = 129).
  - Split: train 1,676 / test 420. Per category (train/test): business 401/100, entertainment 292/73, politics 317/80, sport 400/100, tech 266/67.
  - Words per article (whitespace count, title + text): min 89, median 330, mean 382, max 4,432.
  - `data/articles.csv` columns: doc_id (0..2095), category, title, text, split.
  - Tests: 6 passed (`tests/test_loader.py`).
- **Notes:** In this copy of the dataset, every file decoded as UTF-8, so the latin-1 fallback is kept only as a safety net. No duplicate appears in two different categories. The same-category + same-title rule was added after Step 5 found near-copies in the search results (texts 44–99.95% similar, all checked by hand to be versions of the same story; 9 pairs had been split across train and test). Steps 2–5 outputs were rebuilt after this change. Duplicates are removed *before* doc_id and the split are assigned, so ids have no gaps and no article can be in both train and test. The whitespace word count in `eda.py` is only for the length plot; it is not a tokenizer used for retrieval or classification. The tests read `data/articles.csv`, so run `python -m src.loader` before `pytest`.

## Step 3: Preprocessing

- **Status:** Done
- **Files:** `src/preprocess.py`, `tests/test_preprocess.py`; output `data/tokens.pkl`
- **Key functions (with signatures):**
  - `preprocess(text: str) -> list[str]`: lowercase → `[a-z]+` tokens → drop NLTK English stopwords and one-letter tokens → Porter stem. `STOPWORDS`, `STEMMER` and `TOKEN_PATTERN` are built once at module level; stopwords are downloaded if missing.
  - `tokenize_all() -> dict[int, list[str]]`: `preprocess(title + " " + text)` for every article, keyed by doc_id; saves `data/tokens.pkl`
  - `load_tokens() -> dict[int, list[str]]`: reads `data/tokens.pkl` (later steps use this)
- **Results:**
  - 2,096 articles tokenized; total 453,140 tokens.
  - **Vocabulary size: 18,955 distinct stems.**
  - **Average length: 216.2 tokens per article** (vs 382 raw whitespace words in Step 2).
  - Top 20 stems: said 6788, year 3149, mr 2861, would 2414, also 1998, us 1866, peopl 1835, new 1829, one 1800, time 1570, game 1554, say 1480, could 1408, use 1384, last 1327, make 1320, first 1271, two 1209, govern 1207, go 1168.
  - Tests: 4 new, all 10 tests pass.
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
  - All 2,096 articles (train + test) indexed: N = 2,096, vocabulary 18,955 terms, 453,140 tokens, avgdl = 216.19.
  - df("elect") = 222; first 10 postings (doc_id, tf): (6, 2), (34, 1), (75, 1), (100, 1), (101, 2), (105, 6), (106, 2), (120, 1), (133, 1), (135, 1).
  - 298,012 postings entries in total, vs 2,096 × 18,955 ≈ 39.7 million cells in a full term-document matrix (only 0.75% non-zero). 7,999 terms occur in just one article.
  - Boolean examples: "election AND labour" → 142 docs; "election OR labour" → 304; "election NOT labour" → 80 (142 + 80 = 222 = df("elect")).
  - No document has doc_norm = 0. Tests: 7 new, all 17 pass.
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
  - Toy query ["apple", "iphone"]: TF-IDF D1 0.3061 / D2 0.0367; BM25 (k1=1.2, b=0.75) D1 1.4980 / D2 0.4853; LM JM (λ=0.1) D1 −2.8835 / D2 −6.2930. D3 is never returned. Tests: 6 new, all 23 pass.
  - Demo output (`python -m src.demo_search`):

```
=== Query: "interest rates bank" ===
--- TF-IDF ---
1.    0.3659  [business] Bank set to leave rates on hold  (doc 136)
2.    0.3422  [business] Bank voted 8-1 for no rate change  (doc 150)
3.    0.2869  [business] Australia rates at four year high  (doc 100)
4.    0.2764  [business] House prices show slight increase  (doc 152)
5.    0.2549  [business] Economy 'strong' in election year  (doc 213)
--- BM25 ---
1.   13.5736  [business] Bank set to leave rates on hold  (doc 136)
2.   13.3533  [business] Bank voted 8-1 for no rate change  (doc 150)
3.   13.2564  [business] Bank holds interest rate at 4.75%  (doc 287)
4.   13.0884  [business] UK interest rates held at 4.75%  (doc 430)
5.   12.6377  [business] Australia rates at four year high  (doc 100)
--- Unigram LM ---
1.  -16.4796  [business] Bank holds interest rate at 4.75%  (doc 287)
2.  -16.5927  [business] UK interest rates held at 4.75%  (doc 430)
3.  -16.8768  [business] US interest rates increased to 2%  (doc 120)
4.  -16.9265  [business] Bank opts to leave rates on hold  (doc 118)
5.  -16.9725  [business] Bank voted 8-1 for no rate change  (doc 150)

=== Query: "chelsea premiership" ===
--- TF-IDF ---
1.    0.4519  [sport] Newcastle line up Babayaro  (doc 1535)
2.    0.3866  [sport] Beckham relief as Real go through  (doc 1497)
3.    0.3817  [sport] Ferguson rues failure to cut gap  (doc 1510)
4.    0.3805  [sport] Gallas sees two-horse race  (doc 1468)
5.    0.3792  [sport] Cudicini misses Carling Cup final  (doc 1434)
--- BM25 ---
1.   12.4593  [sport] Houllier praises Benitez regime  (doc 1487)
2.   11.7562  [sport] Cup holders Man Utd visit Everton  (doc 1443)
3.   11.3650  [sport] Newcastle line up Babayaro  (doc 1535)
4.   11.0984  [sport] Ferguson rues failure to cut gap  (doc 1510)
5.   11.0364  [sport] Thompson says Gerrard should stay  (doc 1437)
--- Unigram LM ---
1.  -12.3481  [sport] Houllier praises Benitez regime  (doc 1487)
2.  -12.5523  [sport] Chelsea denied by James heroics  (doc 1399)
3.  -12.8788  [sport] Chelsea hold Arsenal  (doc 1504)
4.  -12.9104  [sport] Desailly backs Blues revenge trip  (doc 1441)
5.  -13.0953  [sport] Cup holders Man Utd visit Everton  (doc 1443)

=== Query: "mobile phone games" ===
--- TF-IDF ---
1.    0.4089  [tech] Home phones face unclear future  (doc 1869)
2.    0.3730  [tech] Half of UK's mobiles 'go online'  (doc 2011)
3.    0.3644  [tech] Mobile gig aims to rock 3G  (doc 1934)
4.    0.3519  [business] Brussels raps mobile call charges  (doc 112)
5.    0.3480  [tech] Mobile gaming takes off in India  (doc 1944)
--- BM25 ---
1.   14.6889  [tech] Moving mobile improves golf swing  (doc 1815)
2.   14.3560  [tech] Mobile audio enters new dimension  (doc 2046)
3.   14.2427  [tech] New consoles promise big problems  (doc 2091)
4.   14.1967  [tech] Cheaper chip for mobiles  (doc 1811)
5.   14.0275  [tech] Mobile gaming takes off in India  (doc 1944)
--- Unigram LM ---
1.  -14.8233  [tech] New consoles promise big problems  (doc 2091)
2.  -15.3169  [tech] Mobile games come of age  (doc 1855)
3.  -15.3442  [tech] 'Friends fear' with lost mobiles  (doc 1829)
4.  -15.5005  [tech] Finding new homes for old phones  (doc 1779)
5.  -15.6102  [tech] Mobile music challenges 'iPod age'  (doc 2083)
```
- **Notes:**
  - TF-IDF divides by the document norm only, not by the query norm. The query norm is the same for every document of one query, so the ranking is identical to full cosine. Full cosine on the toy example would give D1 0.60 / D2 0.07.
  - Only candidates (docs containing at least one query term) are scored. LM query terms with cf = 0 are skipped. Repeated query terms count more than once in BM25 and LM; TF-IDF uses `1 + log10 qtf`.
  - The toy BM25 test uses k1=1.2; the project default is k1=1.5.
  - **Near-duplicates (resolved):** the first demo run showed the same title twice ("Desailly backs Blues revenge trip", "Mobile games come of age"): near-copies that the exact-match rule in Step 2 missed (31 pairs, 9 split across train/test). Step 2 now also drops same-category + same-title repeats; Steps 2–5 outputs were rebuilt and the demo above is the rebuilt run (no repeated titles).

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
  - Trained on the 1,676 train articles only (vocabulary 17,187 stems), tested on the 420 test articles.
  - **Test accuracy: 97.14%** (408/420). Macro F1 0.9709.
  - Per class (precision / recall / F1): business 0.990 / 0.940 / 0.964; entertainment 0.986 / 0.973 / 0.979; politics 0.929 / 0.988 / 0.958; sport 1.000 / 0.980 / 0.990; tech 0.943 / 0.985 / 0.964.
  - Confusion matrix (rows true, columns predicted; order business, entertainment, politics, sport, tech): [94 0 3 0 3], [0 71 2 0 0], [0 0 79 0 1], [1 0 1 98 0], [0 1 0 0 66].
  - Top 15 words per class:
    - business: yuko, oil, gm, glazer, sharehold, rosneft, fiat, gazprom, ebber, worldcom, yugansk, boe, currenc, investor, khodorkovski
    - entertainment: actress, oscar, singer, nomine, rapper, ballet, actor, nomin, foxx, dicaprio, album, eastwood, sideway, swank, scorses
    - politics: tori, lib, blunkett, asylum, ukip, dem, kilroy, blair, forsyth, conserv, kennedi, prescott, milburn, detent, elector
    - sport: roddick, refere, cup, athen, rugbi, seed, gerrard, nadal, mourinho, champion, chelsea, championship, wenger, driscol, henman
    - tech: nintendo, gamer, blog, spam, xbox, portabl, virus, browser, ds, microsoft, user, gadget, devic, handset, softwar
  - IIR Example 13.1 reproduced: P(Chinese|c) = 3/7, P(Tokyo|j) = 2/9, test doc → c. Tests: 7 new, all 30 pass.
- **Notes:** Accuracy is inside the expected 95–98% band, so there is no sign of leakage (near-duplicates were removed in Step 2) or of a preprocessing problem. The 12 errors are mostly genuinely mixed articles: "Ferdinand casts doubt over Glazer" (sport → business: glazer, stake, sharehold), "News Corp eyes video games market" (business → tech: game, video), "Web radio takes Spanish rap global" (tech → entertainment: rap, hip, hop), "Call to overhaul UK state pension" (business → politics). Most errors still get probability ≈ 1.000: Naïve Bayes probabilities are over-confident because the independence assumption counts every correlated word as separate evidence. Only training vocabulary words are used at prediction time, so unseen test words are skipped.

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
  - **BM25: average overlap 9.7 / 10, top result matches on 10 / 10 queries.** Seven queries overlap 10/10; "oil prices economy", "oscar nominations best actress" and "world cup rugby" overlap 9/10 (one document swapped near rank 10).
  - Re-scoring with Okapi's idf gives a max score difference ≤ 3.6e-15 over all 2,096 docs for every query, so the tf, k1, b and length-normalisation parts are identical and idf is the only difference.
  - idf comparison (mine / Okapi raw / Okapi used): said (df 1781) 0.163 / −1.731 / 1.554; bank (179) 2.458 / 2.369 / 2.369; chelsea (66) 3.451 / 3.419 / 3.419; asylum (27) 4.334 / 4.321 / 4.321.
  - **Naïve Bayes: my accuracy 97.14%, scikit-learn MultinomialNB 97.14%, predictions agree on 420 / 420 test articles (100.00%).**
  - Both thresholds (NB agreement ≥ 99%, BM25 overlap ≥ 8/10) pass, so no fixes were needed. All 30 tests still pass.
- **Notes:** The idf difference is the "+ 1" inside my log (always positive) vs Okapi's raw log, which is negative for df > N/2 and is then floored to 0.25 × average idf. None of the 10 queries has a term with df > N/2, so the floor did not cause any of the differences. The swaps come from the "+ 1" giving common words relatively more weight ("world", df 639: 1.188 mine vs 0.824 Okapi; "best", df 368). The floor matters for very common words: "said" (in 85% of articles) gets idf 1.554 in Okapi, close to "price" (2.063), but 0.163 in mine. The Windows console may show "Naïve" as "Na�ve" when printing; `reports/validation.md` itself is UTF-8 and correct.

## Step 8: Query pool

- **Status:** Not started
- **Files:**
- **Key functions (with signatures):**
- **Results:**
- **Notes:**

## Step 9: Evaluation

- **Status:** Not started
- **Files:**
- **Key functions (with signatures):**
- **Results:**
- **Notes:**

## Step 10: App

- **Status:** Not started
- **Files:**
- **Key functions (with signatures):**
- **Results:**
- **Notes:**

## Step 11: Report

- **Status:** Not started
- **Files:**
- **Key functions (with signatures):**
- **Results:**
- **Notes:**
