"""Unit tests for submission validator."""

import pytest
from src.inference.submission_validator import (
    validate_tsv_file,
    MATCHING_HEADER,
    CANDIDATE_HEADER,
)


def test_validate_valid_matching_tsv(tmp_path):
    """Valid matching TSV must parse without errors."""
    tsv_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-10,S3-20\n"
        "S1-002\t\n"
    )
    p = tmp_path / "matching_results.tsv"
    p.write_text(tsv_content, encoding="utf-8")

    errors = []
    mapping = validate_tsv_file(
        str(p),
        MATCHING_HEADER,
        required_s1_ids={"S1-001", "S1-002"},
        col_name="matched_entity_ids",
        errors=errors,
    )
    assert len(errors) == 0
    assert mapping["S1-001"] == {"S2-10", "S3-20"}
    assert mapping["S1-002"] == set()


def test_reject_self_match(tmp_path):
    """Self-match (S1 matching S1) must be rejected."""
    tsv_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS1-002\n"
    )
    p = tmp_path / "matching_results.tsv"
    p.write_text(tsv_content, encoding="utf-8")

    errors = []
    validate_tsv_file(
        str(p),
        MATCHING_HEADER,
        required_s1_ids={"S1-001"},
        col_name="matched_entity_ids",
        errors=errors,
    )
    assert any("self-matches" in e for e in errors)


def test_reject_csv_instead_of_tsv(tmp_path):
    """Comma separated file instead of TSV must be rejected."""
    csv_content = (
        "source1_entity_id,matched_entity_ids\n"
        "S1-001,S2-10\n"
    )
    p = tmp_path / "matching_results.csv"
    p.write_text(csv_content, encoding="utf-8")

    errors = []
    validate_tsv_file(
        str(p),
        MATCHING_HEADER,
        required_s1_ids={"S1-001"},
        col_name="matched_entity_ids",
        errors=errors,
    )
    assert any("header has no TAB" in e for e in errors)
