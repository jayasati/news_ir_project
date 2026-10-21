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

- **Status:** Not started
- **Files:**
- **Key functions (with signatures):**
- **Results:**
- **Notes:**

## Step 3: Preprocessing

- **Status:** Not started
- **Files:**
- **Key functions (with signatures):**
- **Results:**
- **Notes:**

## Step 4: Index

- **Status:** Not started
- **Files:**
- **Key functions (with signatures):**
- **Results:**
- **Notes:**

## Step 5: Rankers

- **Status:** Not started
- **Files:**
- **Key functions (with signatures):**
- **Results:**
- **Notes:**

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
