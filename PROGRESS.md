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
  - `remove_duplicates(df: pd.DataFrame) -> pd.DataFrame`: drops exact (title, text) repeats, keeps the first; prints counts per category
  - `add_split(df: pd.DataFrame) -> pd.DataFrame`: adds split = train/test (stratified 80/20, random_state=42)
  - `load_articles() -> pd.DataFrame`: reads `data/articles.csv` (all later steps use this)
  - `src/eda.py`: `plot_class_distribution(df, path) -> None`, `plot_doc_length(df, path) -> None`
- **Results:**
  - Duplicates removed: business 510→503 (−7), entertainment 386→369 (−17), politics 417→403 (−14), sport 511→505 (−6), tech 401→347 (−54); total 2,225→2,127 (−98).
  - Split: train 1,701 / test 426. Per category (train/test): business 402/101, entertainment 295/74, politics 322/81, sport 404/101, tech 278/69.
  - Words per article (whitespace count, title + text): min 89, median 331, mean 384, max 4,432.
  - `data/articles.csv` columns: doc_id (0..2126), category, title, text, split.
  - Tests: 6 passed (`tests/test_loader.py`).
- **Notes:** In this copy of the dataset, every file decoded as UTF-8, so the latin-1 fallback is kept only as a safety net. No duplicate appears in two different categories. Duplicates are removed *before* doc_id and the split are assigned, so ids have no gaps and no article can be in both train and test. The whitespace word count in `eda.py` is only for the length plot; it is not a tokenizer used for retrieval or classification. The tests read `data/articles.csv`, so run `python -m src.loader` before `pytest`.

## Step 3: Preprocessing

- **Status:** Done
- **Files:** `src/preprocess.py`, `tests/test_preprocess.py`; output `data/tokens.pkl`
- **Key functions (with signatures):**
  - `preprocess(text: str) -> list[str]`: lowercase → `[a-z]+` tokens → drop NLTK English stopwords and one-letter tokens → Porter stem. `STOPWORDS`, `STEMMER` and `TOKEN_PATTERN` are built once at module level; stopwords are downloaded if missing.
  - `tokenize_all() -> dict[int, list[str]]`: `preprocess(title + " " + text)` for every article, keyed by doc_id; saves `data/tokens.pkl`
  - `load_tokens() -> dict[int, list[str]]`: reads `data/tokens.pkl` (later steps use this)
- **Results:**
  - 2,127 articles tokenized; total 463,020 tokens.
  - **Vocabulary size: 18,980 distinct stems.**
  - **Average length: 217.7 tokens per article** (vs 384 raw whitespace words in Step 2).
  - Top 20 stems: said 6930, year 3194, mr 2900, would 2472, also 2037, peopl 1896, us 1885, new 1877, one 1851, game 1620, time 1609, say 1519, use 1457, could 1442, last 1359, make 1347, first 1298, two 1235, govern 1227, like 1195.
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
  - All 2,127 articles (train + test) indexed: N = 2,127, vocabulary 18,980 terms, 463,020 tokens, avgdl = 217.69.
  - df("elect") = 228; first 10 postings (doc_id, tf): (6, 2), (34, 1), (75, 1), (100, 1), (101, 2), (105, 6), (106, 2), (120, 1), (133, 1), (135, 1).
  - 303,945 postings entries in total, vs 2,127 × 18,980 ≈ 40.4 million cells in a full term-document matrix (only 0.75% non-zero). 7,833 terms occur in just one article.
  - Boolean examples: "election AND labour" → 148 docs; "election OR labour" → 310; "election NOT labour" → 80 (148 + 80 = 228 = df("elect")).
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
1.    0.3654  [business] Bank set to leave rates on hold  (doc 136)
2.    0.3423  [business] Bank voted 8-1 for no rate change  (doc 150)
3.    0.2867  [business] Australia rates at four year high  (doc 100)
4.    0.2761  [business] House prices show slight increase  (doc 152)
5.    0.2560  [business] Economy 'strong' in election year  (doc 213)
--- BM25 ---
1.   13.5742  [business] Bank set to leave rates on hold  (doc 136)
2.   13.3547  [business] Bank voted 8-1 for no rate change  (doc 150)
3.   13.2601  [business] Bank holds interest rate at 4.75%  (doc 287)
4.   13.0919  [business] UK interest rates held at 4.75%  (doc 431)
5.   12.6446  [business] Australia rates at four year high  (doc 100)
--- Unigram LM ---
1.  -16.4822  [business] Bank holds interest rate at 4.75%  (doc 287)
2.  -16.5958  [business] UK interest rates held at 4.75%  (doc 431)
3.  -16.8817  [business] US interest rates increased to 2%  (doc 120)
4.  -16.9315  [business] Bank opts to leave rates on hold  (doc 118)
5.  -16.9736  [business] Bank voted 8-1 for no rate change  (doc 150)

=== Query: "chelsea premiership" ===
--- TF-IDF ---
1.    0.4481  [sport] Newcastle line up Babayaro  (doc 1549)
2.    0.3831  [sport] Beckham relief as Real go through  (doc 1511)
3.    0.3784  [sport] Ferguson rues failure to cut gap  (doc 1524)
4.    0.3779  [sport] Gallas sees two-horse race  (doc 1481)
5.    0.3759  [sport] Cudicini misses Carling Cup final  (doc 1446)
--- BM25 ---
1.   12.4249  [sport] Houllier praises Benitez regime  (doc 1500)
2.   11.7240  [sport] Cup holders Man Utd visit Everton  (doc 1456)
3.   11.3297  [sport] Newcastle line up Babayaro  (doc 1549)
4.   11.0667  [sport] Ferguson rues failure to cut gap  (doc 1524)
5.   11.0057  [sport] Thompson says Gerrard should stay  (doc 1449)
--- Unigram LM ---
1.  -12.3430  [sport] Houllier praises Benitez regime  (doc 1500)
2.  -12.5500  [sport] Chelsea denied by James heroics  (doc 1411)
3.  -12.8772  [sport] Chelsea hold Arsenal  (doc 1518)
4.  -12.9086  [sport] Desailly backs Blues revenge trip  (doc 1453)
5.  -12.9086  [sport] Desailly backs Blues revenge trip  (doc 1454)

=== Query: "mobile phone games" ===
--- TF-IDF ---
1.    0.3978  [tech] Home phones face unclear future  (doc 1886)
2.    0.3641  [tech] Half of UK's mobiles 'go online'  (doc 2030)
3.    0.3541  [tech] Mobile gig aims to rock 3G  (doc 1951)
4.    0.3407  [business] Brussels raps mobile call charges  (doc 112)
5.    0.3388  [tech] Mobile gaming takes off in India  (doc 1961)
--- BM25 ---
1.   14.4893  [tech] Moving mobile improves golf swing  (doc 1832)
2.   14.1704  [tech] Mobile audio enters new dimension  (doc 2067)
3.   14.0663  [tech] New consoles promise big problems  (doc 2122)
4.   14.0046  [tech] Cheaper chip for mobiles  (doc 1828)
5.   13.8490  [tech] Mobile gaming takes off in India  (doc 1961)
--- Unigram LM ---
1.  -14.7859  [tech] New consoles promise big problems  (doc 2122)
2.  -15.2696  [tech] Mobile games come of age  (doc 1872)
3.  -15.2900  [tech] Mobile games come of age  (doc 2062)
4.  -15.3016  [tech] 'Friends fear' with lost mobiles  (doc 1846)
5.  -15.4548  [tech] Finding new homes for old phones  (doc 1796)
```
- **Notes:**
  - TF-IDF divides by the document norm only, not by the query norm. The query norm is the same for every document of one query, so the ranking is identical to full cosine. Full cosine on the toy example would give D1 0.60 / D2 0.07.
  - Only candidates (docs containing at least one query term) are scored. LM query terms with cf = 0 are skipped. Repeated query terms count more than once in BM25 and LM; TF-IDF uses `1 + log10 qtf`.
  - The toy BM25 test uses k1=1.2; the project default is k1=1.5.
  - **Near-duplicates:** the demo shows the same title twice ("Desailly backs Blues revenge trip" 1453/1454; "Mobile games come of age" 1872/2062). These texts differ by a paragraph or by whitespace (98.5–99.95% similar), so the exact-match de-duplication in Step 2 kept both. 31 titles are shared by 62 articles; none crosses categories, but 9 pairs are split across train/test. Not changed in this step; still open.

## Step 6: Naïve Bayes

- **Status:** Not started
- **Files:**
- **Key functions (with signatures):**
- **Results:**
- **Notes:**

## Step 7: Validation

- **Status:** Not started
- **Files:**
- **Key functions (with signatures):**
- **Results:**
- **Notes:**

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
