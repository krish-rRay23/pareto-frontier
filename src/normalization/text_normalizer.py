"""Multi-view business name normalizer supporting multilingual text, legal forms, and n-grams."""

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
    "pvt ltd", "private limited", "limited liability", "limited liability company",
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
    "mgmt": "management",
    "svcs": "services",
    "svc": "service",
    "grp": "group",
    "ent": "enterprise",
    "ind": "industries",
    "hldg": "holdings",
}


def strip_accents(text: str) -> str:
    """Normalize accented characters to ASCII equivalents while preserving combining marks safely."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_basic(text: str) -> str:
    """Lowercase, strip accents, and replace punctuation with spaces."""
    if not text or not isinstance(text, str):
        return ""
    text = strip_accents(text).lower()
    text = re.sub(r"[\W_]+", " ", text)
    return text.strip()


def extract_compact(text: str) -> str:
    """Extract alphanumeric only representation (no spaces or symbols)."""
    if not text:
        return ""
    cleaned = strip_accents(str(text)).lower()
    return re.sub(r"[^a-z0-9]", "", cleaned)


def extract_dba(text: str) -> str:
    """Extract trade name if DBA (doing business as) pattern exists."""
    if not text:
        return text
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

    text_joined = " ".join(tokens)
    for term in ["private limited", "pvt ltd", "limited liability company", "limited liability"]:
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
    s = re.sub(r"^[@#*]+", "", text.strip())
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return clean_basic(s)


def extract_acronym(text: str) -> str:
    """Extract acronym from tokens (e.g. 'Amazon Web Services' -> 'aws')."""
    tokens = clean_basic(text).split()
    sig_tokens = [t for t in tokens if t not in LEGAL_TERMS and len(t) >= 2]
    if len(sig_tokens) >= 2:
        return "".join(t[0] for t in sig_tokens[:6])
    return ""


def get_multi_view_name(name: str) -> Dict[str, object]:
    """Generate comprehensive multi-view representation of a business name."""
    from src.normalization.transliteration import has_indic_script, romanize_indic_text

    raw = str(name).strip() if name is not None else ""
    unicode_nfkc = unicodedata.normalize("NFKC", raw)
    cleaned = clean_basic(raw)
    compact = extract_compact(raw)
    dba = clean_basic(extract_dba(raw))
    legal_stripped = strip_legal_terms(raw)
    expanded = expand_abbreviations(legal_stripped)
    social_cleaned = clean_social_and_symbols(raw)

    # Core tokens
    tokens = [t for t in legal_stripped.split() if len(t) >= 1]
    first_two = " ".join(tokens[:2]) if len(tokens) >= 2 else (tokens[0] if tokens else "")
    first_token = tokens[0] if tokens else ""
    last_token = tokens[-1] if tokens else ""

    # Prefix / suffix
    clean_no_space = cleaned.replace(" ", "")
    prefix_4 = clean_no_space[:4] if len(clean_no_space) >= 4 else clean_no_space
    suffix_4 = clean_no_space[-4:] if len(clean_no_space) >= 4 else clean_no_space

    # Order-invariant sorted tokens signature
    sig_tokens = sorted([t for t in tokens if len(t) >= 2 and t not in LEGAL_TERMS])
    sorted_tokens = "_".join(sig_tokens[:5]) if sig_tokens else ""

    # Character n-grams
    char_3grams: Set[str] = set()
    if len(clean_no_space) >= 3:
        char_3grams = {clean_no_space[i : i + 3] for i in range(len(clean_no_space) - 2)}
    char_4grams: Set[str] = set()
    if len(clean_no_space) >= 4:
        char_4grams = {clean_no_space[i : i + 4] for i in range(len(clean_no_space) - 3)}

    # Numeric tokens in name (e.g. '7 Eleven' -> ['7', '11'])
    raw_nums = re.findall(r"\b\d+\b", raw)
    numeric_tokens = [n.lstrip("0") or "0" for n in raw_nums]

    # Acronym
    acronym = extract_acronym(raw)

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
        "unicode_nfkc": unicode_nfkc,
        "clean": cleaned,
        "compact": compact,
        "dba": dba,
        "legal_stripped": legal_stripped,
        "expanded": expanded,
        "social_cleaned": social_cleaned,
        "first_two": first_two,
        "first_token": first_token,
        "last_token": last_token,
        "prefix_4": prefix_4,
        "suffix_4": suffix_4,
        "tokens": tokens,
        "sorted_tokens": sorted_tokens,
        "char_3grams": char_3grams,
        "char_4grams": char_4grams,
        "numeric_tokens": numeric_tokens,
        "acronym": acronym,
        "is_indic": is_indic,
        "romanized": romanized,
        "romanized_tokens": romanized_tokens,
    }
