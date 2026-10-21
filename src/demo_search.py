"""Step 5 demo: top 5 titles from each ranking model for three example queries.

Run from the project root:  python -m src.demo_search
"""

from src.indexer import InvertedIndex
from src.loader import load_articles
from src.rankers import RANKERS, search

QUERIES = ["interest rates bank", "chelsea premiership", "mobile phone games"]

if __name__ == "__main__":
    index = InvertedIndex.load()
    articles = load_articles().set_index("doc_id")  # look up a title by doc_id

    for query in QUERIES:
        print(f'\n=== Query: "{query}" ===')
        for model in RANKERS:
            print(f"--- {model} ---")
            for rank, (doc_id, score) in enumerate(search(index, query, model, k=5), start=1):
                title = articles.loc[doc_id, "title"]
                category = articles.loc[doc_id, "category"]
                print(f"{rank}. {score:9.4f}  [{category}] {title}  (doc {doc_id})")
