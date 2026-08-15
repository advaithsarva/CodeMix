"""Text cleaning for Telugu-English code-mixed text.

Replaces the CleanText() function that lived in Preprocessing.ipynb. That
version had three problems that mattered:

1. It stripped every non-ASCII character (`[^\\x00-\\x7F]+`), which deletes
   all Telugu script. For a code-mixing corpus that removes the entire point.
2. It took three file paths but was called as `df["text"].apply(CleanText)`,
   passing a single string, so it raised TypeError.
3. It applied the Porter stemmer and WordNet lemmatizer — both English-only —
   to every token, including Telugu ones, producing nonsense.

Here cleaning is a pure string -> string function, script-aware, and stdlib
only so it can be tested without downloading models.
"""

from __future__ import annotations

import re
import string
import unicodedata

# Telugu block: U+0C00-U+0C7F.
TELUGU = re.compile(r"[ఀ-౿]")
LATIN = re.compile(r"[A-Za-z]")

# Patterns removed before tokenizing. Deliberately narrow: each one targets
# something that is noise in any language. Anything that would also match
# Telugu or ordinary English words is not in this list.
NOISE_PATTERNS = [
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),   # email
    re.compile(r"https?://\S+|www\.\S+"),                             # URL
    re.compile(r"<[^>]+>"),                                           # HTML tag
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),                       # IP address
    re.compile(r"\b[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\b"),  # UUID
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),                 # date
    re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),                        # card number
    re.compile(r"[@#]\w+"),                                           # mention / hashtag
    re.compile(r"\[\d+\]"),                                           # citation marker
]

# Zero-width and bidi marks ride along in scraped Indic text and break
# tokenisation without ever being visible.
INVISIBLE = re.compile(r"[​-‏‪-‮﻿]")

_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "‘’“”–—")


def script_of(token: str) -> str:
    """Classify a token as 'telugu', 'latin', 'mixed', or 'other'."""
    has_te = bool(TELUGU.search(token))
    has_la = bool(LATIN.search(token))
    if has_te and has_la:
        return "mixed"
    if has_te:
        return "telugu"
    if has_la:
        return "latin"
    return "other"


def tokenize(text: str) -> list[str]:
    """Whitespace tokenisation with punctuation stripped from token edges.

    Good enough for corpus statistics. It deliberately does not split Telugu
    sandhi or English contractions — that needs a real tokeniser, and the
    metrics here do not depend on it.
    """
    tokens = []
    for raw in text.split():
        tok = raw.translate(_PUNCT_TABLE)
        if tok:
            tokens.append(tok)
    return tokens


def clean_text(text: str, *, drop_digits: bool = True) -> str:
    """Normalise one string of code-mixed text.

    Preserves Telugu script. Returns '' for empty or non-string input rather
    than raising, so it is safe to use with pandas .apply() on a column that
    contains NaN.
    """
    if not isinstance(text, str) or not text:
        return ""

    # NFC keeps Telugu vowel signs attached to their consonants; without it,
    # decomposed forms tokenise into separate meaningless codepoints.
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE.sub("", text)

    for pattern in NOISE_PATTERNS:
        text = pattern.sub(" ", text)

    if drop_digits:
        text = re.sub(r"\d+", " ", text)

    text = text.translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def clean_records(records, *, text_key: str = "text"):
    """Clean a list of {id, text} dicts, dropping any that clean to nothing."""
    out = []
    for rec in records:
        cleaned = clean_text(rec.get(text_key, ""))
        if cleaned:
            out.append({**rec, text_key: cleaned})
    return out
