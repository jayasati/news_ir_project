# Viva Notes: Intelligent News Retrieval and Classification System

## Step 2: Loading the articles and removing duplicates

**What the loader does.** `src/loader.py` reads every `.txt` file in `data/bbc/<category>/`. The folder name is
the article's category, the first non-empty line is its title and the rest is its text. It then removes
duplicates, numbers the articles (`doc_id` 0 to 2,126), marks each one as `train` or `test` and saves everything
to `data/articles.csv`. Every later step reads that CSV through `load_articles()`, so all steps use the same
articles and the same split.

**What we found.** The BBC dataset has 2,225 files, but 129 of them repeat an earlier article, so 2,096 unique
articles remain. Duplicates are removed in two passes:
1. **Exact copies** (same title and same text): 98 articles, most of them in tech (54).
2. **Near-copies** (same category and same title): 31 more. These are earlier and later versions of the same
   news story. For example, one says Glenn Hoddle "will be unveiled" as Wolves manager and the other says he
   "has been unveiled", or one version has an extra paragraph. Their texts differ, so the exact check misses
   them, but they are the same article for our purposes. We first found them in Step 5, when the same title
   appeared twice in the search results, and 9 of the pairs had one copy in train and one in test. We checked
   all 31 pairs by hand before removing them. A shared title in a *different* category is not treated as a duplicate.

**Why duplicates must be removed before splitting.**
- *Data leakage.* If an article and its copy are split at random, one copy can land in the training set and
  the other in the test set. The Naïve Bayes classifier would then be "tested" on an article it has already
  seen during training. It would get that one right by memory, not by learning, so the test accuracy would be
  higher than the true accuracy on new articles. The test set must contain only articles the model has never
  seen.
- *Fair retrieval scores.* In search, two identical articles take up two places in the ranking and count twice
  in IDF and document-frequency statistics. Duplicates would also make relevance judgements and metrics like
  precision@k misleading.
- *Order matters.* If we split first and de-duplicate afterwards, copies already sitting on both sides would
  still leak. Removing duplicates first means each unique article is placed in exactly one split.

**Why a stratified 80/20 split.** "Stratified" means each category is split 80/20 separately, so train and
test have the same category proportions as the whole dataset (for example, tech is about 16% of both).
`random_state=42` fixes the random shuffle, so anyone who reruns the code gets exactly the same split.
Result: 1,676 training and 420 test articles, with every category in both.

**Why the UTF-8 / latin-1 fallback.** Some copies of the BBC files contain bytes that are not valid UTF-8, so
reading them as UTF-8 would crash. latin-1 can decode any byte, so it is a safe fallback. (In our copy, every
file was valid UTF-8, so the fallback was never needed.)

## Step 3: Preprocessing (turning text into terms)

All text in the project, both articles and search queries, goes through one function: `preprocess()` in
`src/preprocess.py`. It does five things in order. Example: `"The elections were held in 2005!"`

1. **Lowercase** → `"the elections were held in 2005!"`. "Election" and "election" should be the same term;
   otherwise a query in lowercase would miss a capitalised word at the start of a sentence.
2. **Keep only letter runs `[a-z]+`** → `the, elections, were, held, in`. This splits the text into tokens and
   throws away punctuation and digits in one step. Numbers like years and scores rarely describe a topic and
   would add thousands of useless terms.
3. **Remove stopwords** (NLTK's English list, 198 words) → `elections, held`. Words like "the", "were", "in"
   appear in almost every article, so they cannot tell articles apart. Removing them makes the index smaller and
   the scores more meaningful.
4. **Drop one-letter tokens**. These are mostly left-overs of punctuation, e.g. "U.K." becomes "u" and "k",
   and "it's" becomes "it" and "s". On their own they carry no meaning.
5. **Porter stemming** → `elect, held`. The stemmer cuts suffixes so that different forms of a word become one
   term: "elections", "election", "elected" → "elect"; "running", "runs" → "run". A query for "election" then
   also matches an article about "elections". Stems need not be real words ("peopl", "govern"); they only need to
   be consistent.

Stopwords are removed before stemming because the NLTK list contains whole words, not stems.

**Results on our 2,096 articles:** 453,140 tokens in total, 216.2 tokens per article on average (down from about
382 raw words), and a vocabulary of 18,955 distinct stems. The most frequent stems are "said", "year", "mr",
"would". They are not on the stopword list, but IDF will give them a low weight later because they appear in
most documents.

**Why queries and documents must use the same function.** Retrieval works by matching query terms to the terms
stored in the inverted index. A match only happens if both sides were turned into terms in exactly the same way.
If documents were stemmed but queries were not, the query "elections" would look for the term "elections",
but the index only contains "elect", so nothing would be found. The same goes for lowercasing and stopwords.
Using one shared `preprocess()` guarantees that the query "elections" and an article containing "Elections"
both become "elect" (this is checked by a test). For the same reason, the Naïve Bayes classifier uses this
function too, so the words it learns from are the same as the words it sees when classifying.

**Why save the tokens.** Preprocessing every article takes a few seconds. We do it once, save the result in
`data/tokens.pkl`, and later steps (index, rankers, Naïve Bayes) load it with `load_tokens()`. This also
guarantees every step works with exactly the same terms.

## Step 4: The inverted index

**What it stores.** An inverted index works like the index at the back of a textbook. Instead of listing the
words in each article ("article → words"), it lists, for each word, the articles that contain it
("word → articles"). For every term our `InvertedIndex` keeps a **postings list**: a dictionary
`{doc_id: tf}`, where tf (term frequency) is how many times the term appears in that article. For example,
`postings["elect"]` starts with `{6: 2, 34: 1, 75: 1, ...}`, meaning article 6 contains "elect" twice.

Alongside the postings it stores the numbers the ranking formulas need:
- `df(term)`, the **document frequency**: how many articles contain the term (the length of its postings list).
  "elect" has df = 222. Rare terms (low df) are better at telling articles apart, which is what IDF measures.
- `cf[term]`, the **collection frequency**: how many times the term occurs in the whole collection. The language
  model uses it as the "background" probability of a word.
- `doc_len[doc_id]` (tokens per article), `N` (2,096 articles), `total_tokens` (453,140) and `avgdl`
  (216.2 tokens). BM25 uses these to treat long and short articles fairly.
- `doc_norm[doc_id]`, the length of each article's TF-IDF vector:
  `sqrt( Σ ((1 + log10 tf) · log10(N/df))² )`. TF-IDF cosine similarity divides by this number. It never
  changes, so we compute it once at build time rather than for every query.

The index covers all articles, train and test, because a user should be able to find any article.

**Toy example (from the tests).** D1 = apple launch new iphone, D2 = apple share rise market,
D3 = india win cricket match match. Then `postings["apple"] = {D1: 1, D2: 1}` so df = 2,
`postings["match"] = {D3: 2}`, avgdl = 13/3, and the norm of D1 is
`sqrt(log10(3/2)² + 3 · log10(3)²) ≈ 0.845`.

**Why it is faster than scanning every article.** Without an index, answering "election" means reading all
2,096 articles (about 453,000 words) and checking each one. With the index we do one dictionary lookup,
`postings["elect"]`, and get the 222 matching articles immediately. The work depends on how many articles
contain the query words, not on the size of the whole collection. Most words are rare: in a full
term-by-article table (18,955 × 2,096 ≈ 40 million cells), only 298,012 cells (0.75%) are non-zero. The index
stores just those, so it is also much smaller. The rankers in Step 5 use `candidates()` to score only the
articles that contain at least one query term, and skip the rest.

**Boolean search.** Each query is passed through `preprocess()` first, so it matches the stems in the index.
- AND = intersection of postings sets: "election labour" → 142 articles containing both words.
- OR = union: 304 articles containing either word.
- NOT = AND(a) minus OR(b): "election" but not "labour" → 80 articles (142 + 80 = 222, all articles with "elect").
Boolean search only says yes or no; it cannot rank results. That is why we need TF-IDF, BM25 and the language
model next.

**Saving.** The index is built once and saved with pickle to `models/index.pkl`. Later steps load it with
`InvertedIndex.load()` instead of rebuilding it.

## Step 5: Ranking models (TF-IDF, BM25, unigram language model)

Boolean search only says "match" or "no match". A ranking model gives every matching article a score so the
best ones come first. All three models below score only the **candidates**: articles that contain at least
one query word (looked up in the inverted index). Every query first goes through `preprocess()`.

**Worked example** (the toy corpus from the tests): D1 = apple launch new iphone, D2 = apple share rise market,
D3 = india win cricket match match. N = 3, avgdl = 13/3 ≈ 4.33, query = "apple iphone". D3 contains neither
word, so it is never scored or returned.

### 1. TF-IDF with cosine similarity

Each word gets a weight in the query and in the article:

    w(t, d) = (1 + log10 tf) × idf(t),      idf(t) = log10(N / df(t))
    score(q, d) = Σ_t w(t, q) × w(t, d) / |d|,      |d| = sqrt(Σ w(t, d)²)   (doc_norm from Step 4)

- **tf** (term frequency): a word used more often in an article matters more. The log stops it from growing
  too fast: 10 mentions count about 2×, not 10×.
- **idf** (inverse document frequency): a rare word matters more than a common one. "apple" is in 2 of 3 docs:
  idf = log10(3/2) = 0.176. "iphone" is in 1 of 3: idf = log10(3) = 0.477.
- **Dividing by |d|** turns the dot product into a cosine, i.e. it measures the angle between the query and
  article vectors, so long articles don't win just because they contain more words.

Example: D1 = (0.176 × 0.176 + 0.477 × 0.477) / 0.845 = 0.2587 / 0.845 = **0.306**.
D2 only has "apple": 0.031 / 0.845 = **0.037**.
(Full cosine also divides by the query length |q| = 0.509. That number is the same for every article of one
query, so it can't change the order. We leave it out, and the scores are just scaled by a constant.)

### 2. BM25 (Okapi BM25)

    score(q, d) = Σ_t idf(t) × tf × (k1 + 1) / (tf + k1 × (1 − b + b × |d| / avgdl))
    idf(t) = ln((N − df + 0.5) / (df + 0.5) + 1)

- **k1 (default 1.5)** controls *saturation*. The tf part can never exceed k1 + 1, so the 10th mention of a
  word adds much less than the 1st. k1 = 0 ignores tf completely.
- **b (default 0.75)** controls *length normalisation*. It compares the article's length to the average:
  longer than average lowers the score, shorter raises it. b = 0 ignores length; b = 1 normalises fully.
- The "+ 1" inside the log keeps idf positive even for a word in more than half the articles.

Example with k1 = 1.2, b = 0.75 (the values in the test): D1 has 4 tokens, so the length part is
1 − 0.75 + 0.75 × 4 / 4.33 = 0.942. The tf part for tf = 1 is 1 × 2.2 / (1 + 1.2 × 0.942) = 1.032.
idf(apple) = ln(1.5/2.5 + 1) = ln 1.6 = 0.470; idf(iphone) = ln(2.5/1.5 + 1) = 0.981.
D1 = 1.032 × (0.470 + 0.981) = **1.498**; D2 = 1.032 × 0.470 = **0.485**.

### 3. Query-likelihood unigram language model

Idea: treat each article as a "bag of words" that a writer draws words from. Rank articles by how likely
they are to have produced the query: P(q | d) = Π_t P(t | d). Multiplying many small probabilities would
underflow to 0 on a computer, so we add natural logs instead: **score = Σ_t ln P(t | d)**. All scores are
negative; the one closest to 0 is best.

The plain estimate P(t | d) = tf / |d| is 0 for any query word missing from the article, and ln 0 = −∞,
so an article missing one query word could never be returned. **Smoothing** mixes in the collection model
P(t | C) = cf / total_tokens (how common the word is in all articles):

    Jelinek-Mercer:  P(t | d) = (1 − λ) × tf / |d| + λ × P(t | C)          λ = 0.1
    Dirichlet:       P(t | d) = (tf + μ × P(t | C)) / (|d| + μ)              μ = 2000

Jelinek-Mercer uses a fixed mix. Dirichlet smooths short articles more and long articles less: μ acts like
μ "pretend" words from the collection added to every article. Dirichlet is the project default.

Example with Jelinek-Mercer, λ = 0.1: P(apple | C) = 2/13, P(iphone | C) = 1/13.
D1: ln(0.9 × 1/4 + 0.1 × 2/13) + ln(0.9 × 1/4 + 0.1 × 1/13) = ln 0.2404 + ln 0.2327 = **−2.884**.
D2 has no "iphone": ln 0.2404 + ln(0 + 0.1 × 1/13) = −1.426 − 4.868 = **−6.293**. The missing word costs a lot
but does not make the score −∞.
(With Dirichlet μ = 2000 the toy scores are almost equal, −4.431 vs −4.438, because μ is huge compared with
4-word documents. On real articles, about 216 tokens long, μ = 2000 is a sensible size.)
Query words that never occur anywhere (cf = 0) are skipped, since they would have probability 0 everywhere.

### Why BM25 usually beats plain TF-IDF

1. **Term-frequency saturation.** In TF-IDF, extra mentions keep adding score through 1 + log10 tf with no
   upper limit. In BM25 the tf part levels off at k1 + 1, which matches how relevance really works: an
   article that mentions "chelsea" 20 times is not 20 times more about Chelsea than one that mentions it 5 times.
2. **Tunable length normalisation.** Cosine normalises by vector length in a fixed way, which tends to favour
   very short articles. BM25 compares each article to the *average* length and lets b decide how strong the
   penalty is.
3. **A probabilistic idf**, derived from the probability that a word appears in relevant vs non-relevant
   articles, which down-weights very common words more strongly.
4. **Evidence:** BM25 has been the standard strong baseline in IR research (TREC) for decades, and the
   k1/b defaults work well across many collections. We measure this on our own queries in Step 9.

You can see the difference in our demo. For "interest rates bank", TF-IDF ranks "House prices show slight
increase" 4th, while BM25 and the LM put "Bank holds interest rate at 4.75%" and "UK interest rates held at
4.75%" near the top. Those are more clearly about the query.

## Step 6: Naïve Bayes classifier

**What it does.** Given an article, it predicts its category (business, entertainment, politics, sport or
tech). It learns from the 1,676 training articles only and is tested on the 420 test articles it has never
seen. **Test accuracy: 97.14%** (408 of 420 correct).

**The model (multinomial Naïve Bayes).** Bayes' rule says P(c | d) ∝ P(c) × P(d | c). The "naïve" assumption
is that the words of an article are independent of each other once we know the class, so P(d | c) is just
the product of the word probabilities:

    P(c | d) ∝ P(c) × Π_{t in d} P(t | c)

Training only needs counting:

    P(c)     = N_c / N                                  (share of training articles in class c)
    P(t | c) = (T_ct + 1) / (Σ_t' T_ct' + |V|)           (add-one smoothing, alpha = 1)

T_ct is how many times word t appears in all training articles of class c, and |V| is the number of distinct
words in the training set (17,187).

**Textbook check (IIR Example 13.1).** Training: "Chinese Beijing Chinese", "Chinese Chinese Shanghai",
"Chinese Macao" (class c) and "Tokyo Japan Chinese" (class j). |V| = 6. Class c has 8 words, 5 of them
"Chinese", so P(Chinese | c) = (5 + 1)/(8 + 6) = 3/7. Class j has 3 words, so P(Tokyo | j) = (1 + 1)/(3 + 6) = 2/9.
For "Chinese Chinese Chinese Tokyo Japan": c scores 3/4 × (3/7)³ × (1/14)² ≈ 0.0003 and j scores
1/4 × (2/9)⁵ ≈ 0.0001, so the answer is c. Our code reproduces these numbers in the tests.

**Why add-one smoothing is needed.** Without it, P(t | c) = 0 for any word never seen in class c. One such word
makes the whole product 0, whatever the other 200 words say. In the example, "Tokyo" never appears in class c,
so c would score 0 and lose, even though "Chinese" appears three times. Adding 1 to every count gives unseen
words a small probability (1/14 here) instead of 0. This is the same idea as smoothing in the language model.

**Why we work with logs.** An article has about 216 words, each with a probability around 0.001 or less. Their
product is roughly 10⁻⁶⁰⁰, far below the smallest number a computer can store (about 10⁻³⁰⁸), so it would
become 0 for every class (underflow). Taking logs turns the product into a sum:

    score(c) = log P(c) + Σ_{t in d} log P(t | c)

Sums of negative numbers around −2,000 are no problem. Because log is increasing, the class with the highest
log score is also the class with the highest probability.

To show probabilities, `predict_proba` uses a **stable softmax**: subtract the largest score m before exponentiating,
P(c | d) = exp(score_c − m) / Σ exp(score_c' − m). Without subtracting m, exp(−2000) would underflow to 0 for
every class and we would divide 0 by 0.

**Explaining a prediction.** `top_words(c)` lists the words whose probability in c is highest compared with the
best other class (log-likelihood ratio). For politics these are "tori" (Tory), "lib", "blunkett", "asylum",
"ukip"; for sport "roddick", "refere", "cup", "rugbi"; for tech "nintendo", "gamer", "blog", "spam".
`explain(doc, c)` shows which words in one article pushed it towards c rather than the runner-up class.

**Where it goes wrong.** Most of the 12 errors are genuinely mixed articles. "Ferdinand casts doubt over Glazer"
is a sport article about a takeover, so words like "stake" and "sharehold" made it look like business. "Web
radio takes Spanish rap global" is tech, but "rap" and "hip hop" point to entertainment. Naïve Bayes still gives
most of these wrong answers a probability close to 1.000. Its probabilities are over-confident because the
independence assumption counts related words ("hip" and "hop") as separate pieces of evidence. So we trust
its predicted label, not the exact probability.

**How Naïve Bayes relates to the unigram language model.** They are the same model used in two directions.
- In Step 5, each *article* has a unigram language model P(t | d), and we rank articles by how likely they are to
  generate the *query*: Σ log P(t | d).
- In Naïve Bayes, each *class* has a unigram language model P(t | c), built by pooling all its training articles,
  and we choose the class most likely to generate the *article*: log P(c) + Σ log P(t | c).
Both assume words are drawn independently (bag of words), both sum log probabilities, and both need smoothing
for unseen words. Naïve Bayes uses add-one (Laplace) smoothing; the retrieval LM uses Dirichlet or
Jelinek-Mercer, mixing in the collection model. The only extra piece in Naïve Bayes is the prior P(c).
