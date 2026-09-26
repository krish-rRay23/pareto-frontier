"""Address normalization and multi-view token extractor."""

import re
from typing import Dict, List, Set
from src.normalization.text_normalizer import strip_accents, clean_basic


STREET_ABBREVIATIONS: Dict[str, str] = {
    # English
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
    # French
    "r": "rue",
    "av": "avenue",
    "bd": "boulevard",
    "imp": "impasse",
    "all": "allee",
    "pl": "place",
    "chem": "chemin",
}

ADDR_STOPWORDS: Set[str] = {
    "st", "street", "rd", "road", "dr", "drive", "ave", "avenue", "ln", "lane",
    "blvd", "boulevard", "ct", "court", "cir", "circle", "hwy", "highway",
    "fl", "floor", "ste", "suite", "apt", "apartment", "unit", "po", "box",
    "near", "opp", "opposite", "behind", "beside", "block", "sector", "plot",
    "no", "h", "city", "state", "road", "west", "east", "north", "south",
    # French stopwords
    "rue", "boulevard", "avenue", "impasse", "allee", "place", "chemin", "batiment",
}


def extract_numbers(text: str) -> List[str]:
    """Extract all sequences of 1-7 digits with leading zeros stripped."""
    if not text or not isinstance(text, str):
        return []
    raw_nums = re.findall(r"\b\d{1,7}\b", text)
    return [n.lstrip("0") or "0" for n in raw_nums]


def extract_postal_codes(text: str) -> List[str]:
    """Extract 5-digit (US ZIP / France code) or 6-digit (India PIN) postal codes."""
    if not text or not isinstance(text, str):
        return []
    # US 5-digit / France 5-digit: \b\d{5}\b
    # India 6-digit PIN: \b[1-9]\d{5}\b
    return re.findall(r"\b\d{5,6}\b", text)


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


def extract_house_number(text: str) -> str:
    """Extract candidate house/building number (typically the first number)."""
    nums = extract_numbers(text)
    return nums[0] if nums else ""


def get_multi_view_address(address: str) -> Dict[str, object]:
    """Generate multi-view representation of a business address."""
    raw = str(address).strip() if address is not None else ""
    norm = normalize_address(raw)
    cleaned = clean_basic(raw)
    numbers = extract_numbers(raw)
    postal_codes = extract_postal_codes(raw)
    salient = extract_salient_tokens(raw)

    first_num = numbers[0] if numbers else ""
    second_num = numbers[1] if len(numbers) >= 2 else ""
    first_postal = postal_codes[0] if postal_codes else ""
    first_salient = salient[0] if salient else ""
    second_salient = salient[1] if len(salient) >= 2 else ""

    tokens = norm.split()
    token_set = set(tokens)

    # Character 3-grams
    clean_no_space = cleaned.replace(" ", "")
    char_3grams: Set[str] = set()
    if len(clean_no_space) >= 3:
        char_3grams = {clean_no_space[i : i + 3] for i in range(len(clean_no_space) - 2)}

    return {
        "raw": raw,
        "clean": cleaned,
        "norm": norm,
        "tokens": tokens,
        "token_set": token_set,
        "numbers": numbers,
        "number_set": set(numbers),
        "first_num": first_num,
        "second_num": second_num,
        "postal_codes": postal_codes,
        "first_postal": first_postal,
        "salient": salient,
        "salient_set": set(salient),
        "first_salient": first_salient,
        "second_salient": second_salient,
        "char_3grams": char_3grams,
        "is_empty": (len(raw) == 0),
    }
