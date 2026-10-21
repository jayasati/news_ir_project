# Viva Notes: Intelligent News Retrieval and Classification System

## Step 2: Loading the articles and removing duplicates

**What the loader does.** `src/loader.py` reads every `.txt` file in `data/bbc/<category>/`. The folder name is
the article's category, the first non-empty line is its title and the rest is its text. It then removes
duplicates, numbers the articles (`doc_id` 0 to 2,126), marks each one as `train` or `test` and saves everything
to `data/articles.csv`. Every later step reads that CSV through `load_articles()`, so all steps use the same
articles and the same split.

**What we found.** The BBC dataset has 2,225 files, but 98 of them repeat an earlier article word for word (same
title and same text). Most are in tech (54). After removing them, 2,127 unique articles remain.

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
Result: 1,701 training and 426 test articles, with every category in both.

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

**Results on our 2,127 articles:** 463,020 tokens in total, 217.7 tokens per article on average (down from about
384 raw words), and a vocabulary of 18,980 distinct stems. The most frequent stems are "said", "year", "mr",
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
  "elect" has df = 228. Rare terms (low df) are better at telling articles apart, which is what IDF measures.
- `cf[term]`, the **collection frequency**: how many times the term occurs in the whole collection. The language
  model uses it as the "background" probability of a word.
- `doc_len[doc_id]` (tokens per article), `N` (2,127 articles), `total_tokens` (463,020) and `avgdl`
  (217.7 tokens). BM25 uses these to treat long and short articles fairly.
- `doc_norm[doc_id]`, the length of each article's TF-IDF vector:
  `sqrt( Σ ((1 + log10 tf) · log10(N/df))² )`. TF-IDF cosine similarity divides by this number. It never
  changes, so we compute it once at build time rather than for every query.

The index covers all articles, train and test, because a user should be able to find any article.

**Toy example (from the tests).** D1 = apple launch new iphone, D2 = apple share rise market,
D3 = india win cricket match match. Then `postings["apple"] = {D1: 1, D2: 1}` so df = 2,
`postings["match"] = {D3: 2}`, avgdl = 13/3, and the norm of D1 is
`sqrt(log10(3/2)² + 3 · log10(3)²) ≈ 0.845`.

**Why it is faster than scanning every article.** Without an index, answering "election" means reading all
2,127 articles (about 463,000 words) and checking each one. With the index we do one dictionary lookup,
`postings["elect"]`, and get the 228 matching articles immediately. The work depends on how many articles
contain the query words, not on the size of the whole collection. Most words are rare: in a full
term-by-article table (18,980 × 2,127 ≈ 40 million cells), only 303,945 cells (0.75%) are non-zero. The index
stores just those, so it is also much smaller. The rankers in Step 5 use `candidates()` to score only the
articles that contain at least one query term, and skip the rest.

**Boolean search.** Each query is passed through `preprocess()` first, so it matches the stems in the index.
- AND = intersection of postings sets: "election labour" → 148 articles containing both words.
- OR = union: 310 articles containing either word.
- NOT = AND(a) minus OR(b): "election" but not "labour" → 80 articles (148 + 80 = 228, all articles with "elect").
Boolean search only says yes or no; it cannot rank results. That is why we need TF-IDF, BM25 and the language
model next.

**Saving.** The index is built once and saved with pickle to `models/index.pkl`. Later steps load it with
`InvertedIndex.load()` instead of rebuilding it.
