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
