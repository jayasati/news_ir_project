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
