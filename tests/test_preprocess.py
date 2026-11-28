"""
Tests for src/preprocess.py (small fake data, run in about a second).
Run from the project root:  pytest tests/test_preprocess.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import preprocess  # noqa: E402


# ------------------------------------------------------------------ 1. clean_text
def test_clean_text_replaces_literal_newlines_and_spaces():
    raw = "Dear Team,\\n\\nThe   server\tis down."
    assert preprocess.clean_text(raw, True, True) == "Dear Team, The server is down."


def test_clean_text_missing_value_becomes_empty():
    assert preprocess.clean_text(float("nan"), True, True) == ""


# ------------------------------------------------------------------ 2. build_joined_text
def test_joined_text_with_and_without_subject():
    assert preprocess.build_joined_text("Billing", "Wrong invoice", ". ") == "Billing. Wrong invoice"
    assert preprocess.build_joined_text("", "Wrong invoice", ". ") == "Wrong invoice"


# ------------------------------------------------------------------ 3. normalize_text
def test_normalize_text_ignores_case_and_punctuation():
    first = preprocess.normalize_text("Several products are impacted.")
    second = preprocess.normalize_text("several  products are IMPACTED")
    assert first == second == "several products are impacted"


# ------------------------------------------------------------------ 4. Union-Find
def test_union_find_families():
    parent = [0, 1, 2, 3, 4, 5]
    preprocess.union_groups(parent, 0, 1)
    preprocess.union_groups(parent, 1, 2)
    preprocess.union_groups(parent, 4, 5)
    assert preprocess.union_groups(parent, 2, 0) is False          # already one family
    assert preprocess.find_root(parent, 0) == preprocess.find_root(parent, 2)
    assert preprocess.find_root(parent, 3) == 3
    assert preprocess.find_root(parent, 4) == preprocess.find_root(parent, 5)
    assert preprocess.find_root(parent, 0) != preprocess.find_root(parent, 4)


# ------------------------------------------------------------------ 5-6. group split
def make_fake_tickets():
    """60 families of 1-3 tickets in 3 departments."""
    rows = []
    ticket_number = 0
    for family_number in range(60):
        department = ["A", "B", "C"][family_number % 3]
        family_size = 1 + family_number % 3
        for _ in range(family_size):
            rows.append({"ticket_id": ticket_number, "queue": department, "family_id": family_number})
            ticket_number = ticket_number + 1
    return pd.DataFrame(rows)


SHARES = {"train": 0.70, "validation": 0.15, "test": 0.15}


def test_group_split_keeps_every_family_in_one_split():
    tickets = make_fake_tickets()
    tickets["split"] = preprocess.stratified_group_split(tickets, "queue", SHARES, random_seed=42)
    splits_per_family = tickets.groupby("family_id")["split"].nunique()
    assert (splits_per_family == 1).all()
    assert set(tickets["split"]) == {"train", "validation", "test"}


def test_group_split_is_repeatable_with_the_same_seed():
    tickets = make_fake_tickets()
    first = preprocess.stratified_group_split(tickets, "queue", SHARES, random_seed=42)
    second = preprocess.stratified_group_split(tickets, "queue", SHARES, random_seed=42)
    assert first == second
