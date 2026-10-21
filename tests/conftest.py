"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def toy_docs() -> dict[int, list[str]]:
    """A tiny corpus whose tokens are ALREADY processed (do not run preprocess() on them).

    doc_id 1 = D1, 2 = D2, 3 = D3. Total 13 tokens, so avgdl = 13/3.
    """
    return {
        1: ["apple", "launch", "new", "iphone"],
        2: ["apple", "share", "rise", "market"],
        3: ["india", "win", "cricket", "match", "match"],
    }
