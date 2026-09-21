"""High-performance catalog and text feature extraction framework."""

import re
from typing import Any, Dict

import numpy as np
import pandas as pd

# Standard Unit Conversion Factors
MASS_TO_GRAMS = {
    "mg": 0.001,
    "g": 1.0,
    "gm": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kgs": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "oz": 28.3495,
    "ounce": 28.3495,
    "ounces": 28.3495,
    "lb": 453.592,
    "lbs": 453.592,
    "pound": 453.592,
    "pounds": 453.592,
}

VOLUME_TO_ML = {
    "ml": 1.0,
    "milliliter": 1.0,
    "millilitre": 1.0,
    "millilitres": 1.0,
    "milliliters": 1.0,
    "l": 1000.0,
    "ltr": 1000.0,
    "litre": 1000.0,
    "liter": 1000.0,
    "litres": 1000.0,
    "liters": 1000.0,
    "fl oz": 29.5735,
    "fluid ounce": 29.5735,
    "gallon": 3785.41,
}

# Regex compilation for high throughput
RE_PACK_COUNT = re.compile(
    r"(?:pack\s+of\s*(\d+)|set\s+of\s*(\d+)|combo\s+of\s*(\d+)|(\d+)\s*(?:pack|pk|pcs|pieces|count|units?|sets?|pairs?))",
    re.IGNORECASE,
)

RE_MULTIPLIER = re.compile(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

ALL_UNITS = (
    list(MASS_TO_GRAMS.keys())
    + list(VOLUME_TO_ML.keys())
    + ["count", "pcs", "piece", "pieces", "units", "tablets", "capsules", "sheets", "sachets"]
)
ALL_UNITS_SORTED = sorted(ALL_UNITS, key=len, reverse=True)
UNITS_REGEX_STR = (
    r"(?<![a-zA-Z])(" + "|".join(re.escape(u) for u in ALL_UNITS_SORTED) + r")(?![a-zA-Z])"
)

RE_NUM_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*" + UNITS_REGEX_STR,
    re.IGNORECASE,
)

RE_BRAND_LABEL = re.compile(
    r"(?:brand\s*:\s*|by\s+)([a-zA-Z0-9\-\.\&]+)",
    re.IGNORECASE,
)

KEYWORD_INDICATORS = [
    "combo",
    "pack",
    "set",
    "organic",
    "refill",
    "premium",
    "pro",
    "plus",
    "mini",
    "max",
    "ultra",
    "wireless",
    "rechargeable",
    "genuine",
    "original",
    "warranty",
    "waterproof",
    "portable",
    "natural",
]


def extract_pack_count(text: str) -> int:
    """Extract pack or set count from text string."""
    if not text or not isinstance(text, str):
        return 1

    # Check for multiplier like '3x500g' or '2 x 100'
    m_mult = RE_MULTIPLIER.search(text)
    if m_mult:
        try:
            return int(m_mult.group(1))
        except (ValueError, IndexError):
            pass

    m_pack = RE_PACK_COUNT.search(text)
    if m_pack:
        for grp in m_pack.groups():
            if grp and grp.isdigit():
                val = int(grp)
                if 1 <= val <= 1000:
                    return val

    return 1


def extract_quantity_and_unit(text: str) -> Dict[str, Any]:
    """Extract primary numeric quantity, unit, and standardized weight/volume."""
    result = {
        "extracted_value": np.nan,
        "extracted_unit": "unknown",
        "norm_weight_g": np.nan,
        "norm_volume_ml": np.nan,
    }
    if not text or not isinstance(text, str):
        return result

    matches = RE_NUM_UNIT.findall(text)
    if not matches:
        return result

    # Pick first matching numerical measurement
    val_str, unit_raw = matches[0]
    try:
        val = float(val_str)
        unit = unit_raw.lower().strip()
        result["extracted_value"] = val
        result["extracted_unit"] = unit

        if unit in MASS_TO_GRAMS:
            result["norm_weight_g"] = val * MASS_TO_GRAMS[unit]
        elif unit in VOLUME_TO_ML:
            result["norm_volume_ml"] = val * VOLUME_TO_ML[unit]
    except Exception:
        pass

    return result


def extract_brand_candidate(text: str) -> str:
    """Heuristic extraction of brand candidate from title or text."""
    if not text or not isinstance(text, str):
        return "UNKNOWN"

    # 1. Look for explicit 'Brand: <brand>' or 'by <brand>'
    m_label = RE_BRAND_LABEL.search(text)
    if m_label:
        return m_label.group(1).strip()

    # 2. Extract first token if title starts with standard capitalized word
    tokens = text.strip().split()
    if tokens:
        first_token = re.sub(r"[^\w\s]", "", tokens[0])
        if (
            first_token
            and (first_token.istitle() or first_token.isupper())
            and len(first_token) > 1
        ):
            return first_token

    return "UNKNOWN"


def compute_text_stats(text: str, prefix: str = "txt") -> Dict[str, float]:
    """Compute text complexity and density metrics."""
    if not text or not isinstance(text, str):
        return {
            f"{prefix}_char_len": 0.0,
            f"{prefix}_word_count": 0.0,
            f"{prefix}_digit_ratio": 0.0,
            f"{prefix}_upper_ratio": 0.0,
            f"{prefix}_punct_ratio": 0.0,
        }

    char_len = float(len(text))
    words = text.split()
    word_count = float(len(words))

    n_digits = sum(c.isdigit() for c in text)
    n_upper = sum(c.isupper() for c in text)
    n_punct = sum(not c.isalnum() and not c.isspace() for c in text)

    denom = max(char_len, 1.0)
    return {
        f"{prefix}_char_len": char_len,
        f"{prefix}_word_count": word_count,
        f"{prefix}_digit_ratio": float(n_digits / denom),
        f"{prefix}_upper_ratio": float(n_upper / denom),
        f"{prefix}_punct_ratio": float(n_punct / denom),
    }


def extract_keyword_indicators(text: str) -> Dict[str, int]:
    """Extract binary presence flags for common value-shifting keywords."""
    t_lower = text.lower() if isinstance(text, str) else ""
    return {f"kw_{kw}": int(kw in t_lower) for kw in KEYWORD_INDICATORS}


def extract_catalog_record_features(
    title: str = "",
    desc: str = "",
    bullets: str = "",
) -> Dict[str, Any]:
    """Extract full feature dictionary from a single catalog product record."""
    combined_text = f"{title} {bullets} {desc}".strip()

    pack_count = extract_pack_count(combined_text)
    qty_info = extract_quantity_and_unit(combined_text)
    brand = extract_brand_candidate(title if title else combined_text)

    # Text statistics per field
    title_stats = compute_text_stats(title, prefix="title")
    desc_stats = compute_text_stats(desc, prefix="desc")
    bullets_stats = compute_text_stats(bullets, prefix="bullets")

    # Keyword indicators
    kw_flags = extract_keyword_indicators(combined_text)

    # Normalized effective quantity
    effective_weight = (
        qty_info["norm_weight_g"] * pack_count
        if not np.isnan(qty_info["norm_weight_g"])
        else np.nan
    )
    effective_vol = (
        qty_info["norm_volume_ml"] * pack_count
        if not np.isnan(qty_info["norm_volume_ml"])
        else np.nan
    )
    effective_raw_qty = (
        qty_info["extracted_value"] * pack_count
        if not np.isnan(qty_info["extracted_value"])
        else np.nan
    )

    features: Dict[str, Any] = {
        "pack_count": pack_count,
        "brand_candidate": brand,
        "extracted_value": qty_info["extracted_value"],
        "extracted_unit": qty_info["extracted_unit"],
        "norm_weight_g": qty_info["norm_weight_g"],
        "norm_volume_ml": qty_info["norm_volume_ml"],
        "effective_weight_g": effective_weight,
        "effective_volume_ml": effective_vol,
        "effective_quantity": effective_raw_qty,
        **title_stats,
        **desc_stats,
        **bullets_stats,
        **kw_flags,
    }

    return features


def extract_features_dataframe(
    df: pd.DataFrame,
    title_col: str = "catalog_title",
    desc_col: str = "description",
    bullets_col: str = "bullet_points",
) -> pd.DataFrame:
    """Vectorized / batch extraction of tabular features for an entire DataFrame."""
    records = []
    titles = (
        df[title_col].fillna("").astype(str).tolist() if title_col in df.columns else [""] * len(df)
    )
    descs = (
        df[desc_col].fillna("").astype(str).tolist() if desc_col in df.columns else [""] * len(df)
    )
    bullets = (
        df[bullets_col].fillna("").astype(str).tolist()
        if bullets_col in df.columns
        else [""] * len(df)
    )

    for t, d, b in zip(titles, descs, bullets, strict=False):
        records.append(extract_catalog_record_features(t, d, b))

    return pd.DataFrame(records, index=df.index)
