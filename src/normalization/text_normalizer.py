"""Multi-view business name normalizer supporting multilingual text and legal stripping."""

import re
import unicodedata
from typing import Dict, List, Set


LEGAL_TERMS: Set[str] = {
    # US / UK / Common law
    "llc", "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "pllc", "pc", "co", "company", "lp", "llp", "gmbh",
    # France / Civil law
    "sarl", "sasu", "sas", "eurl", "sa", "snc", "scs", "sci", "gie",
    # India common variants
    "pvt ltd", "private limited", "limited liability",
}

ABBREVIATIONS: Dict[str, str] = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "co": "company",
    "mfg": "manufacturing",
    "intl": "international",
    "assoc": "associates",
    "tech": "technology",
}


def strip_accents(text: str) -> str:
    """Normalize accented characters to ASCII equivalents (e.g. Ó -> O, é -> e)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_basic(text: str) -> str:
    """Lowercase, strip accents, and replace punctuation with spaces."""
    if not text or not isinstance(text, str):
        return ""
    text = strip_accents(text).lower()
    # Replace symbols and punctuation with space
    text = re.sub(r"[\W_]+", " ", text)
    return text.strip()


def extract_dba(text: str) -> str:
    """Extract trade name if DBA (doing business as) pattern exists."""
    if not text:
        return text
    # Matches patterns like 'Company A d/b/a Brand B' or 'Company A DBA Brand B'
    m = re.search(r"\b(?:d\s*/?\s*b\s*/?\s*a|t\s*/?\s*a)\b[:\s]*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def strip_legal_terms(text: str) -> str:
    """Strip legal entity suffixes, preserving core business name."""
    cleaned = clean_basic(text)
    tokens = cleaned.split()
    if not tokens:
        return ""
    
    # Check multi-word legal terms
    text_joined = " ".join(tokens)
    for term in ["private limited", "pvt ltd", "limited liability company"]:
        if text_joined.endswith(term):
            text_joined = text_joined[: -len(term)].strip()
            tokens = text_joined.split()

    filtered = [t for t in tokens if t not in LEGAL_TERMS]
    return " ".join(filtered) if filtered else " ".join(tokens)


def expand_abbreviations(text: str) -> str:
    """Expand common business abbreviations."""
    tokens = clean_basic(text).split()
    expanded = [ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(expanded)


def clean_social_and_symbols(text: str) -> str:
    """Clean social handles, leading @/#, and split concatenated patterns."""
    if not text:
        return ""
    # Strip leading @ or #
    s = re.sub(r"^[@#]+", "", text.strip())
    # Split camelCase if present (e.g. PrimeMoney -> Prime Money)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return clean_basic(s)


def get_multi_view_name(name: str) -> Dict[str, object]:
    """Generate multi-view representation of a business name."""
    from src.normalization.transliteration import has_indic_script, romanize_indic_text

    raw = str(name).strip() if name is not None else ""
    cleaned = clean_basic(raw)
    dba = clean_basic(extract_dba(raw))
    legal_stripped = strip_legal_terms(raw)
    expanded = expand_abbreviations(legal_stripped)
    social_cleaned = clean_social_and_symbols(raw)

    # Core tokens (first 2 significant tokens)
    tokens = [t for t in legal_stripped.split() if len(t) >= 2]
    first_two = " ".join(tokens[:2]) if len(tokens) >= 2 else (tokens[0] if tokens else "")

    # Order-invariant sorted tokens signature
    sig_tokens = sorted([t for t in tokens if len(t) >= 3])
    sorted_tokens = "_".join(sig_tokens[:4]) if sig_tokens else ""

    # Indic script romanization
    is_indic = has_indic_script(raw)
    romanized = ""
    romanized_tokens = []
    if is_indic:
        rom_raw = romanize_indic_text(raw)
        romanized = strip_legal_terms(rom_raw)
        romanized_tokens = [t for t in romanized.split() if len(t) >= 2]

    return {
        "raw": raw,
        "clean": cleaned,
        "dba": dba,
        "legal_stripped": legal_stripped,
        "expanded": expanded,
        "social_cleaned": social_cleaned,
        "first_two": first_two,
        "tokens": tokens,
        "sorted_tokens": sorted_tokens,
        "is_indic": is_indic,
        "romanized": romanized,
        "romanized_tokens": romanized_tokens,
    }

