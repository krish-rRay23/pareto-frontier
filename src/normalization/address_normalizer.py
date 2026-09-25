"""Address normalization and entity resolution token extractor."""

import re
from typing import Dict, List, Set
from src.normalization.text_normalizer import strip_accents, clean_basic


STREET_ABBREVIATIONS: Dict[str, str] = {
    "rd": "road",
    "st": "street",
    "dr": "drive",
    "ave": "avenue",
    "ln": "lane",
    "blvd": "boulevard",
    "bd": "boulevard",
    "ct": "court",
    "cir": "circle",
    "hwy": "highway",
    "pkwy": "parkway",
    "ste": "suite",
    "fl": "floor",
    "apt": "apartment",
    "bldg": "building",
    "pl": "place",
    "sq": "square",
    "ter": "terrace",
}

US_STATE_CODES: Dict[str, str] = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

ADDR_STOPWORDS: Set[str] = {
    "st", "street", "rd", "road", "dr", "drive", "ave", "avenue", "ln", "lane",
    "blvd", "boulevard", "ct", "court", "cir", "circle", "hwy", "highway",
    "fl", "floor", "ste", "suite", "apt", "apartment", "unit", "po", "box",
    "near", "opp", "opposite", "behind", "beside", "block", "sector", "plot",
    "no", "h", "city", "state", "road", "west", "east", "north", "south",
}


def extract_numbers(text: str) -> List[str]:
    """Extract all sequences of 1-6 digits with leading zeros stripped (e.g. 0017560 -> 17560)."""
    if not text or not isinstance(text, str):
        return []
    raw_nums = re.findall(r"\b\d{1,7}\b", text)
    # Normalize by stripping leading zeros so '005559' matches '5559'
    return [n.lstrip("0") or "0" for n in raw_nums]


def extract_postal_codes(text: str) -> List[str]:
    """Extract 5-digit (US ZIP) or 6-digit (India PIN) postal codes."""
    if not text or not isinstance(text, str):
        return []
    return re.findall(r"\b[1-9]\d{4,5}\b", text)


def normalize_address(text: str) -> str:
    """Normalize street types, casing, accents, and spacing."""
    cleaned = clean_basic(text)
    tokens = cleaned.split()
    normalized = [STREET_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(normalized)


def extract_salient_tokens(text: str, min_len: int = 3) -> List[str]:
    """Extract distinctive address tokens (excluding stop-words and pure digits)."""
    cleaned = clean_basic(text)
    tokens = cleaned.split()
    salient = [
        t for t in tokens
        if len(t) >= min_len and t not in ADDR_STOPWORDS and not t.isdigit()
    ]
    return salient


def get_multi_view_address(address: str) -> Dict[str, object]:
    """Generate multi-view representation of a business address."""
    raw = str(address).strip() if address is not None else ""
    norm = normalize_address(raw)
    numbers = extract_numbers(raw)
    postal_codes = extract_postal_codes(raw)
    salient = extract_salient_tokens(raw)
    
    first_num = numbers[0] if numbers else ""
    first_salient = salient[0] if salient else ""
    second_salient = salient[1] if len(salient) >= 2 else ""

    # Sort salient tokens for order-invariant address matching
    sorted_salient = sorted(salient)

    return {
        "raw": raw,
        "clean": norm,
        "numbers": numbers,
        "postal_codes": postal_codes,
        "primary_number": first_num,
        "salient_tokens": salient,
        "sorted_salient": sorted_salient,
        "primary_salient": first_salient,
        "secondary_salient": second_salient,
    }

