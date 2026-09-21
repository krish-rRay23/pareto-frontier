"""Unit tests for catalog text parsers and feature extractors."""

from src.features.text_extraction import (
    compute_text_stats,
    extract_brand_candidate,
    extract_catalog_record_features,
    extract_keyword_indicators,
    extract_pack_count,
    extract_quantity_and_unit,
)


def test_extract_pack_count():
    """Test regex extraction of various pack and set formats."""
    assert extract_pack_count("Dabur Honey 500g (Pack of 3)") == 3
    assert extract_pack_count("Set of 4 Stainless Steel Bowls") == 4
    assert extract_pack_count("6 Pack Athletic Socks") == 6
    assert extract_pack_count("2x500g Peanut Butter") == 2
    assert extract_pack_count("Single bottle shampoo") == 1
    assert extract_pack_count("") == 1


def test_extract_quantity_and_unit():
    """Test unit normalization and weight/volume conversion."""
    res_g = extract_quantity_and_unit("Premium Almonds 500g Bag")
    assert res_g["extracted_value"] == 500.0
    assert res_g["extracted_unit"] == "g"
    assert res_g["norm_weight_g"] == 500.0

    res_kg = extract_quantity_and_unit("Basmati Rice 5 kg pack")
    assert res_kg["extracted_value"] == 5.0
    assert res_kg["extracted_unit"] == "kg"
    assert res_kg["norm_weight_g"] == 5000.0

    res_ml = extract_quantity_and_unit("Olive Oil 750ml")
    assert res_ml["extracted_value"] == 750.0
    assert res_ml["extracted_unit"] == "ml"
    assert res_ml["norm_volume_ml"] == 750.0

    res_l = extract_quantity_and_unit("Mineral Water 2 L bottle")
    assert res_l["extracted_value"] == 2.0
    assert res_l["norm_volume_ml"] == 2000.0


def test_extract_brand_candidate():
    """Test brand candidate heuristics."""
    assert extract_brand_candidate("Samsung Galaxy S24 Ultra") == "Samsung"
    assert extract_brand_candidate("Wireless Mouse by Logitech") == "Logitech"
    assert extract_brand_candidate("Brand: Philips Series 3000 Trimmer") == "Philips"
    assert extract_brand_candidate("") == "UNKNOWN"


def test_compute_text_stats():
    """Test character and token complexity metrics."""
    stats = compute_text_stats("Hello World 123!", prefix="t")
    assert stats["t_char_len"] == 16.0
    assert stats["t_word_count"] == 3.0
    assert stats["t_digit_ratio"] > 0
    assert stats["t_upper_ratio"] > 0
    assert stats["t_punct_ratio"] > 0


def test_extract_keyword_indicators():
    """Test presence of marketing and packaging keywords."""
    flags = extract_keyword_indicators("Organic Green Tea Combo Pack with Warranty")
    assert flags["kw_organic"] == 1
    assert flags["kw_combo"] == 1
    assert flags["kw_pack"] == 1
    assert flags["kw_warranty"] == 1
    assert flags["kw_wireless"] == 0


def test_extract_catalog_record_features():
    """Test combined record feature dictionary."""
    feats = extract_catalog_record_features(
        title="Nestle Coffee 200g (Pack of 2)",
        desc="Pure rich blend",
        bullets="Net: 200g | 2 Pack",
    )
    assert feats["pack_count"] == 2
    assert feats["brand_candidate"] == "Nestle"
    assert feats["effective_weight_g"] == 400.0
