"""High-speed native Indic script romanization and transliteration.

Maps Brahmic Unicode characters (Devanagari, Bengali, Gujarati, Gurmukhi,
Oriya, Tamil, Telugu, Kannada, Malayalam) to Latin phonemes.
Also handles social handle symbols and joined prefixes (@, #, d/b/a).
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple

# All Brahmic scripts in Unicode follow identical layout offset % 0x80
# 0x0900 (Devanagari) to 0x0D7F (Malayalam)
BRAHMIC_OFFSET_MAP: Dict[int, str] = {
    # Vowels
    0x02: "m", 0x03: "h",
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au",
    # Consonants
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "ch", 0x1B: "ch", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    # Matras (vowel signs)
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x4D: "", # Virama / halant (suppress inherent vowel)
}

NON_ASCII_INDIC_PATTERN = re.compile(r"[\u0900-\u0D7F]")
ZERO_WIDTH_CHARS = {"\u200c", "\u200d", "\ufeff"}


def has_indic_script(text: str) -> bool:
    """Check if string contains any Indic Unicode characters."""
    if not text or not isinstance(text, str):
        return False
    return bool(NON_ASCII_INDIC_PATTERN.search(text))


def romanize_indic_text(text: str) -> str:
    """Transliterate Indic Brahmic text to Latin phonemes."""
    if not text or not isinstance(text, str):
        return ""
    
    out: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x0D7F:
            offset = code % 0x80
            mapped = BRAHMIC_OFFSET_MAP.get(offset, "")
            out.append(mapped)
        elif ch in ZERO_WIDTH_CHARS:
            continue
        else:
            out.append(ch)
            
    res = "".join(out)
    # Collapse duplicate characters from matras/phonemes (e.g. 'ee' or 'ss')
    res = re.sub(r"([a-z])\1{2,}", r"\1\1", res)
    return res
