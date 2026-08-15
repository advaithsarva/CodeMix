"""Loading and quality auditing for the code-mixed corpus.

The bundled data/ files are 1M records shaped like {"id": int, "text": str},
but sampling them shows sequences such as:

    "or సమయం or కాబట్టి and పూర్తి message message చాలా తో కాని తో సమయం task."

which is not code-mixed language — it is randomly sampled Telugu and English
tokens. Metrics computed on it are meaningless, so this module measures the
properties that distinguish real code-mixed text from shuffled tokens rather
than leaving it to eyeballing.

Stdlib only, so the audit runs without transformers installed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from preprocessing import script_of, tokenize

# Function words carry grammar. Real sentences use them at a characteristic
# rate and in varied positions; a shuffled bag reuses a handful at random.
EN_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "is", "are", "was", "were", "be", "been", "it", "this",
    "that", "as", "by", "from", "so", "than", "then",
}


def load_corpus(path, limit=None):
    """Read a JSON array of records. `limit` reads only the first N."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array, got {type(data).__name__}")
    return data[:limit] if limit else data


def code_mix_index(tokens) -> float:
    """Code-Mixing Index (CMI) for one utterance, 0-100.

    The standard measure for code-mixed text (Das & Gamback). 0 means the
    utterance is entirely one language; higher means a more even mix.
    Tokens belonging to no language (numbers, symbols) are excluded.
    """
    langs = [script_of(t) for t in tokens]
    counted = [l for l in langs if l in ("telugu", "latin", "mixed")]
    n = len(counted)
    if n == 0:
        return 0.0

    freq = Counter(counted)
    max_lang = max(freq.values())
    return 100.0 * (1 - max_lang / n)


def adjacent_repeat_rate(tokens) -> float:
    """Fraction of positions where a token equals the one before it.

    Real text almost never repeats a token immediately. Randomly sampling
    from a vocabulary does it at roughly 1/vocab_size per position, and the
    bundled data does it far more often than natural text.
    """
    if len(tokens) < 2:
        return 0.0
    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return repeats / (len(tokens) - 1)


def audit(records, *, text_key: str = "text", sample: int = 5000) -> dict:
    """Summarise corpus health over the first `sample` records."""
    subset = records[:sample]

    total_tokens = 0
    vocab = Counter()
    cmis = []
    repeat_rates = []
    function_word_hits = 0
    latin_tokens = 0
    empty = 0

    for rec in subset:
        tokens = tokenize(rec.get(text_key, "") or "")
        if not tokens:
            empty += 1
            continue

        total_tokens += len(tokens)
        vocab.update(tokens)
        cmis.append(code_mix_index(tokens))
        repeat_rates.append(adjacent_repeat_rate(tokens))

        for tok in tokens:
            if script_of(tok) == "latin":
                latin_tokens += 1
                if tok.lower() in EN_FUNCTION_WORDS:
                    function_word_hits += 1

    n = len(cmis) or 1
    return {
        "records_examined": len(subset),
        "empty_records": empty,
        "total_tokens": total_tokens,
        "vocabulary_size": len(vocab),
        "type_token_ratio": round(len(vocab) / total_tokens, 4) if total_tokens else 0.0,
        "mean_tokens_per_record": round(total_tokens / n, 2),
        "mean_code_mix_index": round(sum(cmis) / n, 2),
        "mean_adjacent_repeat_rate": round(sum(repeat_rates) / n, 4),
        "function_word_share_of_english": (
            round(function_word_hits / latin_tokens, 4) if latin_tokens else 0.0
        ),
        "most_common_tokens": vocab.most_common(10),
    }


def looks_synthetic(report: dict) -> bool:
    """Heuristic verdict on whether a corpus is shuffled tokens, not language.

    Two independent signals, both far outside what natural text produces:
    immediate token repetition, and an English function-word share that looks
    like uniform sampling rather than grammar (natural English running text
    is roughly 40-50% function words).
    """
    return (
        report["mean_adjacent_repeat_rate"] > 0.01
        or report["function_word_share_of_english"] < 0.15
    )


def format_report(report: dict) -> str:
    lines = ["Corpus audit", "=" * 40]
    for key, value in report.items():
        if key == "most_common_tokens":
            top = ", ".join(f"{t}({c})" for t, c in value)
            lines.append(f"{'most_common_tokens':<32} {top}")
        else:
            lines.append(f"{key:<32} {value}")

    lines.append("=" * 40)
    if looks_synthetic(report):
        lines.append(
            "VERDICT: this corpus does not look like natural code-mixed language.\n"
            "         Token repetition and/or function-word distribution are\n"
            "         inconsistent with real sentences. Any score computed on\n"
            "         it is not meaningful — replace it with a real corpus\n"
            "         (LinCE, GLUECoS, or L3Cube-HingCorpus-style Telugu sets)\n"
            "         before reporting results."
        )
    else:
        lines.append("VERDICT: statistics are consistent with natural text.")
    return "\n".join(lines)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Audit a code-mixed corpus JSON file.")
    parser.add_argument("path", type=Path, help="JSON array of {id, text} records")
    parser.add_argument("--sample", type=int, default=5000,
                        help="records to examine (default: 5000)")
    args = parser.parse_args(argv)

    records = load_corpus(args.path, limit=args.sample)
    print(format_report(audit(records, sample=args.sample)))


if __name__ == "__main__":
    main()
