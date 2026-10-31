from collections import Counter

import pytest

from specdec.prompts import CATEGORIES, build_prompts


def test_balanced_and_deterministic():
    p = build_prompts(16)
    assert len(p) == 16
    assert Counter(c for c, _ in p) == {c: 4 for c in CATEGORIES}
    assert p == build_prompts(16)
    assert len({text for _, text in p}) == 16  # every template used once at n=16


def test_round_robin_order():
    assert [c for c, _ in build_prompts(8)] == list(CATEGORIES) * 2


def test_code_edit_prompts_contain_code():
    for cat, text in build_prompts(16):
        if cat == "code_edit":
            assert "```" in text


def test_copy_categories_have_source_text_to_copy():
    # n-gram lookup needs material in the prompt; these prompts should be long
    for cat, text in build_prompts(16):
        if cat in ("code_edit", "extract", "rag_qa"):
            assert len(text.split()) > 60
        else:
            assert len(text.split()) < 40


def test_validation():
    with pytest.raises(ValueError):
        build_prompts(0)
    with pytest.raises(ValueError):
        build_prompts(4, categories=("nope",))
