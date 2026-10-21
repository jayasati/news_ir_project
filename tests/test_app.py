"""Tests for Step 10 (app.py) using Streamlit's AppTest. Needs models/index.pkl, models/nb.pkl and data files."""

from streamlit.testing.v1 import AppTest


def start_app() -> AppTest:
    # The path is relative to this test file; the app itself reads data relative to the project root,
    # which is where pytest is run from.
    at = AppTest.from_file("../app.py", default_timeout=120)
    at.run()
    return at


def test_app_starts_without_exception():
    at = start_app()
    assert not at.exception
    assert at.title[0].value == "Intelligent News Retrieval and Classification"


def test_search_shows_results():
    at = start_app()
    at.text_input(key="query").input("interest rates bank").run()
    assert not at.exception
    assert any("results with BM25" in c.value for c in at.caption)
    # A known relevant article appears among the result markdown blocks.
    assert any("Bank" in m.value and "score" in m.value for m in at.markdown)


def test_search_with_only_stopwords_warns():
    at = start_app()
    at.text_input(key="query").input("the and of").run()
    assert not at.exception
    assert any("no searchable words" in w.value for w in at.warning)


def test_classify_tab():
    at = start_app()
    at.text_area(key="article_text").input("The striker scored twice as Chelsea beat Arsenal in the Premiership.")
    at.button(key="classify").click().run()
    assert not at.exception
    assert any("sport" in m.value for m in at.markdown)
