"""Unit tests for Entity Resolution normalizers."""

import pytest
from src.normalization.text_normalizer import (
    clean_basic,
    extract_dba,
    get_multi_view_name,
    strip_accents,
    strip_legal_terms,
)
from src.normalization.address_normalizer import (
    extract_numbers,
    extract_salient_tokens,
    get_multi_view_address,
    normalize_address,
)


def test_strip_accents():
    """Accented characters must fold cleanly to ASCII."""
    assert strip_accents("Novent Ówl") == "Novent Owl"
    assert strip_accents("Président") == "President"
    assert strip_accents("Hauts-de-France") == "Hauts-de-France"


def test_strip_legal_terms():
    """Legal suffixes across US, India, and France must be stripped."""
    assert strip_legal_terms("Novent Owl PLLC") == "novent owl"
    assert strip_legal_terms("Prime Money Inc") == "prime money"
    assert strip_legal_terms("Solar Solutions Private Limited") == "solar solutions"
    assert strip_legal_terms("ZNB Club SARL") == "znb club"
    assert strip_legal_terms("Elephant Centre EURL") == "elephant centre"


def test_extract_dba():
    """DBA trade names must be captured."""
    assert extract_dba("Beloavi d/b/a Novent Owl PLLC") == "Novent Owl PLLC"


def test_address_normalization():
    """Street types and formatting should normalize consistently."""
    norm = normalize_address("9308 Home Ct, Des Plaines, IL")
    assert "court" in norm
    assert "9308" in norm


def test_number_extraction():
    """Building and postal numbers should be accurately extracted."""
    nums = extract_numbers("8124, Sector - C, Pocket - 8 Vasant Kunj, New Delhi 110070")
    assert "8124" in nums
    assert "8" in nums
    assert "110070" in nums


def test_salient_address_tokens():
    """Salient address tokens should exclude generic stopwords."""
    salient = extract_salient_tokens("1795 Westchester Drive, High Point, NC")
    assert "westchester" in salient
    assert "drive" not in salient
