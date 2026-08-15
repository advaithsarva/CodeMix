"""NLLB-200 translation for Telugu-English code-mixed text.

Fixes two bugs from the original evaluation.py:

1.  It prefixed the source string with `>>tel_Latn<< `. That is the
    Marian / OPUS-MT convention; NLLB does not parse it and simply
    translated the tag as literal text. NLLB takes the source language from
    `tokenizer.src_lang`, which the old code never set — so every input was
    tokenised as whatever the tokeniser defaulted to.

2.  It used `tel_Latn`, which is romanised Telugu. The corpus is in Telugu
    script, which NLLB calls `tel_Telu`. The code was requesting the wrong
    language even ignoring bug 1.

The model is loaded lazily so the pure helpers below can be imported and
tested without downloading 600M parameters.
"""

from __future__ import annotations

from itertools import islice

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

# The NLLB codes this project actually uses. NLLB tags are always
# <iso639-3>_<ISO 15924 script>, which is what makes tel_Telu vs tel_Latn a
# meaningful distinction rather than a typo.
LANGS = {
    "tel_Telu": "Telugu (Telugu script)",
    "tel_Latn": "Telugu (romanised)",
    "hin_Deva": "Hindi (Devanagari)",
    "eng_Latn": "English",
}

# The corpus is Telugu script, so this is the correct default source.
DEFAULT_SRC = "tel_Telu"
DEFAULT_TGT = "hin_Deva"


def resolve_lang(code: str) -> str:
    """Validate an NLLB language tag, failing loudly on a bad one.

    A wrong-but-plausible tag is the failure mode that started this file:
    it produces confident garbage instead of an error.
    """
    if code in LANGS:
        return code

    known = ", ".join(sorted(LANGS))
    if "_" not in code:
        raise ValueError(
            f"{code!r} is not an NLLB language tag. NLLB tags include the "
            f"script, e.g. 'tel_Telu' not 'te'. Known here: {known}"
        )
    raise ValueError(f"Unknown NLLB language tag {code!r}. Known here: {known}")


def batched(items, size: int):
    """Yield lists of up to `size` items. (itertools.batched needs 3.12.)"""
    if size < 1:
        raise ValueError("batch size must be >= 1")
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


class Translator:
    """Lazy NLLB wrapper. Nothing is downloaded until .translate() is called."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        # Imported here, not at module scope, so `import translate` stays
        # cheap and testable when transformers is absent.
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)
        self._model.eval()

    def translate(
        self,
        sentences,
        src_lang: str = DEFAULT_SRC,
        tgt_lang: str = DEFAULT_TGT,
        batch_size: int = 8,
        max_new_tokens: int = 128,
    ) -> list[str]:
        src = resolve_lang(src_lang)
        tgt = resolve_lang(tgt_lang)

        sentences = [s for s in sentences if s and s.strip()]
        if not sentences:
            return []

        self._load()
        import torch

        # This is the correct NLLB contract: source language on the
        # tokenizer, target language forced as the first generated token.
        self._tokenizer.src_lang = src
        bos = self._tokenizer.convert_tokens_to_ids(tgt)

        out = []
        for batch in batched(sentences, batch_size):
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(self.device)

            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs, forced_bos_token_id=bos, max_new_tokens=max_new_tokens
                )

            out.extend(self._tokenizer.batch_decode(generated, skip_special_tokens=True))
        return out


class Embedder:
    """Lazy multilingual sentence embedder."""

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def encode(self, sentences, batch_size: int = 32):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(list(sentences), batch_size=batch_size)
