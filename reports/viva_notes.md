# Viva Notes: Intelligent News Retrieval and Classification System

## Step 2: Loading the articles and removing duplicates

**What the loader does.** `src/loader.py` reads every `.txt` file in `data/bbc/<category>/`. The folder name is
the article's category, the first non-empty line is its title and the rest is its text. It then removes
duplicates, numbers the articles (`doc_id` 0 to 2,057), marks each one as `train` or `test` and saves everything
to `data/articles.csv`. Every later step reads that CSV through `load_articles()`, so all steps use the same
articles and the same split.

**What we found.** The BBC dataset has 2,225 files, but 167 of them repeat an earlier article, so 2,058 unique
articles remain. Duplicates are removed in four passes, always keeping the first copy:
1. **Exact copies** (same title and same text): 98 articles, most of them in tech (54).
2. **Same category and same title**: 31 more. These are earlier and later versions of the same news story. For
   example, one says Glenn Hoddle "will be unveiled" as Wolves manager and the other says he "has been unveiled",
   or one version has an extra paragraph.
3. **Same category and same text**: 35 more. The story is word for word the same, but the BBC re-published it
   under a reworded headline, e.g. "Tory backing for ID cards" and "Conservative backing for ID cards".
4. **Near-copies**: 3 more. A reworded headline *and* a tiny text change, such as a fixed typo ("Internet
   Electronic" → "Internet Engineering") or a new paragraph break. We compare the *sets of words* in two
   articles of the same category with the **Jaccard similarity** |A ∩ B| / |A ∪ B| (shared words divided by
   all words) and remove a copy when it is at least 0.95. The three removed pairs score 0.985–1.0. The next most
   similar pair scores only 0.85 and is a genuinely updated story ("talks will take place" versus "talks have
   ended"), so it is kept.

We did not find all of these at once. Rules 2–4 were added after we spotted repeated stories in our own search
results (Step 5) and in the query pool (Step 8), and every pair was checked by hand before removal. Before the
fix, 23 of these pairs had one copy in train and one in test. A shared title in a *different* category is never
treated as a duplicate.

**Why duplicates must be removed before splitting.**
- *Data leakage.* If an article and its copy are split at random, one copy can land in the training set and
  the other in the test set. The Naïve Bayes classifier would then be "tested" on an article it has already
  seen during training. It would get that one right by memory, not by learning, so the test accuracy would be
  higher than the true accuracy on new articles. The test set must contain only articles the model has never
  seen. We saw this happen: with rules 3 and 4 missing, accuracy was 97.14%; after removing those 38 copies it
  fell to 96.60%. The higher number was partly the model recognising articles it had already seen.
- *Fair retrieval scores.* In search, two identical articles take up two places in the ranking and count twice
  in IDF and document-frequency statistics. Duplicates would also make relevance judgements and metrics like
  precision@k misleading.
- *Order matters.* If we split first and de-duplicate afterwards, copies already sitting on both sides would
  still leak. Removing duplicates first means each unique article is placed in exactly one split.

**Why a stratified 80/20 split.** "Stratified" means each category is split 80/20 separately, so train and
test have the same category proportions as the whole dataset (for example, tech is about 16% of both).
`random_state=42` fixes the random shuffle, so anyone who reruns the code gets exactly the same split.
Result: 1,646 training and 412 test articles, with every category in both.

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

**Results on our 2,058 articles:** 444,650 tokens in total, 216.1 tokens per article on average (down from about
381 raw words), and a vocabulary of 18,955 distinct stems. The most frequent stems are "said", "year", "mr",
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
  "elect" has df = 215. Rare terms (low df) are better at telling articles apart, which is what IDF measures.
- `cf[term]`, the **collection frequency**: how many times the term occurs in the whole collection. The language
  model uses it as the "background" probability of a word.
- `doc_len[doc_id]` (tokens per article), `N` (2,058 articles), `total_tokens` (444,650) and `avgdl`
  (216.1 tokens). BM25 uses these to treat long and short articles fairly.
- `doc_norm[doc_id]`, the length of each article's TF-IDF vector:
  `sqrt( Σ ((1 + log10 tf) · log10(N/df))² )`. TF-IDF cosine similarity divides by this number. It never
  changes, so we compute it once at build time rather than for every query.

The index covers all articles, train and test, because a user should be able to find any article.

**Toy example (from the tests).** D1 = apple launch new iphone, D2 = apple share rise market,
D3 = india win cricket match match. Then `postings["apple"] = {D1: 1, D2: 1}` so df = 2,
`postings["match"] = {D3: 2}`, avgdl = 13/3, and the norm of D1 is
`sqrt(log10(3/2)² + 3 · log10(3)²) ≈ 0.845`.

**Why it is faster than scanning every article.** Without an index, answering "election" means reading all
2,058 articles (about 445,000 words) and checking each one. With the index we do one dictionary lookup,
`postings["elect"]`, and get the 215 matching articles immediately. The work depends on how many articles
contain the query words, not on the size of the whole collection. Most words are rare: in a full
term-by-article table (18,955 × 2,058 ≈ 39 million cells), only 292,454 cells (0.75%) are non-zero. The index
stores just those, so it is also much smaller. The rankers in Step 5 use `candidates()` to score only the
articles that contain at least one query term, and skip the rest.

**Boolean search.** Each query is passed through `preprocess()` first, so it matches the stems in the index.
- AND = intersection of postings sets: "election labour" → 137 articles containing both words.
- OR = union: 294 articles containing either word.
- NOT = AND(a) minus OR(b): "election" but not "labour" → 78 articles (137 + 78 = 215, all articles with "elect").
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
increase" 4th and "Economy 'strong' in election year" 5th, while BM25 and the LM fill their top 5 with articles
about interest-rate decisions, such as "Bank holds interest rate at 4.75%" and "Bank opts to leave rates on
hold". Those are more clearly about the query.

## Step 6: Naïve Bayes classifier

**What it does.** Given an article, it predicts its category (business, entertainment, politics, sport or
tech). It learns from the 1,646 training articles only and is tested on the 412 test articles it has never
seen. **Test accuracy: 96.60%** (398 of 412 correct).

**The model (multinomial Naïve Bayes).** Bayes' rule says P(c | d) ∝ P(c) × P(d | c). The "naïve" assumption
is that the words of an article are independent of each other once we know the class, so P(d | c) is just
the product of the word probabilities:

    P(c | d) ∝ P(c) × Π_{t in d} P(t | c)

Training only needs counting:

    P(c)     = N_c / N                                  (share of training articles in class c)
    P(t | c) = (T_ct + 1) / (Σ_t' T_ct' + |V|)           (add-one smoothing, alpha = 1)

T_ct is how many times word t appears in all training articles of class c, and |V| is the number of distinct
words in the training set (17,287).

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
"ukip"; for sport "roddick", "cup", "refere", "rugbi"; for tech "blog", "spam", "nintendo", "gamer".
`explain(doc, c)` shows which words in one article pushed it towards c rather than the runner-up class.

**Where it goes wrong.** Most of the 14 errors are genuinely mixed articles. "Ferdinand casts doubt over Glazer"
is a sport article about a takeover of Manchester United, so words like "glazer" and "stake" made it look like
business. "CSI shows give 'unrealistic view'" is entertainment (a TV show), but "lab", "scienc" and "test" point
to tech. Entertainment loses the most (recall 0.917), because arts stories about money, councils or technology
read like politics or tech. Naïve Bayes still gives 13 of the 14 wrong answers a probability above 0.99. Its
probabilities are over-confident because the independence assumption counts related words (like "lab" and
"scienc") as separate pieces of evidence. So we trust its predicted label, not the exact probability.

**How Naïve Bayes relates to the unigram language model.** They are the same model used in two directions.
- In Step 5, each *article* has a unigram language model P(t | d), and we rank articles by how likely they are to
  generate the *query*: Σ log P(t | d).
- In Naïve Bayes, each *class* has a unigram language model P(t | c), built by pooling all its training articles,
  and we choose the class most likely to generate the *article*: log P(c) + Σ log P(t | c).
Both assume words are drawn independently (bag of words), both sum log probabilities, and both need smoothing
for unseen words. Naïve Bayes uses add-one (Laplace) smoothing; the retrieval LM uses Dirichlet or
Jelinek-Mercer, mixing in the collection model. The only extra piece in Naïve Bayes is the prior P(c).

## Step 7: Proving the implementations are correct

Writing BM25 and Naïve Bayes from scratch is only worth something if they are right. We checked them in two
ways: unit tests on small examples with known answers (Steps 4–6), and, in this step, by running well-known
library versions on exactly the same data and comparing the outputs (`src/validate.py`,
results in `reports/validation.md`). The libraries were used only for checking, never inside our system.

**1. BM25 vs `rank_bm25.BM25Okapi`.** Both were given the same token lists (from `tokens.pkl`, in doc_id order)
and the same parameters (k1 = 1.5, b = 0.75). For 10 queries covering all five categories, we compared the
top 10 articles:
- The top result was the same for **all 10 queries**.
- **All 10 of the top 10** results were the same for every query (average overlap 10.0 / 10). (On an earlier
  version of the data, before the last duplicates were removed, three queries matched 9/10 with one article
  swapped near rank 10. The explanation below covers that case.)

**Why they are not 100% identical.** The two versions use a slightly different idf:
- ours: idf = ln((N − df + 0.5)/(df + 0.5) **+ 1**), which is always positive;
- BM25Okapi: idf = ln((N − df + 0.5)/(df + 0.5)), which goes negative for words in more than half the
  articles, and those negative values are replaced by 0.25 × the average idf (the "epsilon floor").

To prove idf is the *only* difference, we re-scored every article with our formula but plugged in
BM25Okapi's idf values. The scores then matched BM25Okapi's to about 0.000000000000004, i.e. exactly, apart from
computer rounding. So our tf saturation, k1, b and length normalisation are implemented correctly. Any swaps
happen because the "+1" gives common words a bit more weight in our version. For "world" (in 630
articles), our idf is 1.183 and Okapi's is 0.818, so an article that mentions "world" a lot can move up one
place near rank 10.

(Side note: the epsilon floor can do odd things to very common words. "said" is in 1,746 of 2,058 articles,
so its raw Okapi idf is −1.721, but the floor raises it to 1.552, while ours gives it 0.165. Our "+1" version
handles common words more sensibly, which is one reason it is the form used in modern search engines such as
Lucene.)

**2. Naïve Bayes vs scikit-learn `MultinomialNB(alpha = 1.0)`.** We trained scikit-learn's model on the same
1,646 training token lists (`CountVectorizer(analyzer=lambda x: x)` just counts the tokens we already have,
without tokenizing again) and predicted the same 412 test articles.
- Our accuracy: **96.60%**; scikit-learn's: **96.60%**.
- The two models made the **same prediction on 412 of 412 test articles (100%)**, including the same 14
  mistakes.

This is expected, because both compute the same thing: log prior + sum of add-one-smoothed log likelihoods,
over the training vocabulary only. Identical predictions on every article show our counting, smoothing,
log-space scoring and argmax are all correct.

**How to say it in the viva:** "I validated my implementations against standard libraries on the same
input. Naïve Bayes agrees with scikit-learn on 100% of test articles. BM25 has the same top result as
rank_bm25 on every query and 100% overlap in the top 10. The formulas differ only in the idf variant, and I
proved it: with their idf, my scores match theirs to floating-point precision."

## Step 8: The query pool (preparing relevance judgements)

**Why we need relevance judgements.** To measure how good a search model is (precision, recall, nDCG in
Step 9), we need to know the *right answers*: for each test query, which articles are actually relevant.
The BBC dataset only has category labels, and a category is not a query. "Chelsea premiership" is a sport
query, but most of the 500 sport articles are about tennis, rugby or athletics and are not relevant to it. So we
wrote 20 realistic queries (`data/queries.csv`, 4 per category, 2–4 words each, never just the category name)
and judge the results ourselves.

**What pooling is.** Judging all 2,058 articles for all 20 queries would mean 41,160 judgements, which is
impossible by hand. Pooling is the standard shortcut used in TREC evaluations:
1. Run every system (here TF-IDF, BM25 and the unigram LM) on every query.
2. Take the top k results of each (k = 10, the "pool depth").
3. Merge them into one set without duplicates: the pool.
4. A human judges only the pooled documents. Anything outside the pool is assumed not relevant.

This works because a relevant article that none of the three models puts in its top 10 cannot change any of
their top-10 scores anyway. Our pools have 11–18 articles per query (289 in total) instead of 30, because the
three models often agree on the same articles.

**Limitation.** Pooling can miss relevant articles that all three models ranked low, so measures that depend
on knowing *every* relevant document (like recall) can be a little too optimistic. Measures on the top 10
(precision@10, nDCG@10) are exact, because every top-10 document has been judged.

**Why I judged relevance by hand.**
- *Category is not relevance.* Category labels would say every sport article is relevant to "chelsea
  premiership", which is wrong, and they can't tell a passing mention from an article that is really about the
  topic.
- *Avoiding circular evaluation.* If relevance were decided by a rule built from the same words the models use
  (for example "contains the query term"), we would be measuring the models against themselves, and they would
  all look perfect. A human reads for *meaning*.
- *Graded relevance.* A person can say how relevant: 0 = not relevant, 1 = relevant, 2 = highly relevant
  (the article is mainly about the query topic). nDCG uses these grades to reward putting the best articles
  first.
- *Fair judging.* The pool is sorted by doc_id, so I could not see which model found an article or at what rank,
  which prevents bias towards one model. The `suggested_rel` column is only a hint (right category and contains
  the rarest query word). It is 1 for 97% of rows, so it cannot replace reading the article.

The judgements become `data/qrels.csv` ("query relevance judgements"), which Step 9 uses to score the models.

**Duplicates found while building the pool.** The first pool showed "Gadget show heralds MP3 season" and
"Gadget show heralds MP3 Christmas" side by side: the same article with two headlines. That led to rules 3 and 4
in Step 2 and a rebuild of everything from the loader to the pool. We did this *before* judging, because removing
articles changes the doc_ids, and judgements made on the old pool would have pointed at the wrong articles.
Judging a pool that still contains copies would also count the same story twice in precision@10.
