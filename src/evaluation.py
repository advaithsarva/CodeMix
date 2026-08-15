"""Translate code-mixed records, and score them when references exist.

Replaces the original script, which hardcoded a single example sentence,
used the wrong NLLB language tag, and passed the language as a text prefix
that the model never interpreted (see translate.py for the details).

Usage:
    python src/evaluation.py --text "మన goals ద్వారా we can do wonders"
    python src/evaluation.py --corpus data/Processed/code1.json --limit 20
    python src/evaluation.py --corpus pairs.json --reference-key hindi
    python src/evaluation.py --hypotheses out.txt --references gold.txt

Scoring is reference-based (chrF++ and BLEU). Without references the script
only translates — there is no meaningful quality number to report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import metrics
from corpus import load_corpus
from preprocessing import clean_text
from translate import DEFAULT_SRC, DEFAULT_TGT, Embedder, Translator


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def semantic_similarity(sources, translations) -> float | None:
    """Mean cosine similarity between source and translation embeddings.

    A reference-free sanity signal: the embedder is multilingual, so a good
    translation should land near its source. It is not a quality metric —
    it cannot tell a fluent wrong answer from a right one — so it is
    reported alongside chrF++, never instead of it.
    """
    embedder = Embedder()
    src_vecs = embedder.encode(sources)
    tgt_vecs = embedder.encode(translations)

    sims = [
        metrics.cosine_similarity(list(map(float, a)), list(map(float, b)))
        for a, b in zip(src_vecs, tgt_vecs)
    ]
    return round(sum(sims) / len(sims), 4) if sims else None


def evaluate(sentences, references=None, src_lang=DEFAULT_SRC, tgt_lang=DEFAULT_TGT,
             embed=True):
    """Translate sentences, then score against references if given."""
    pairs = [(clean_text(s), r) for s, r in zip(sentences, references or [None] * len(sentences))]
    pairs = [(s, r) for s, r in pairs if s]
    if not pairs:
        return {"pairs": [], "scores": {}}

    cleaned = [s for s, _ in pairs]
    refs = [r for _, r in pairs]

    translations = Translator().translate(cleaned, src_lang=src_lang, tgt_lang=tgt_lang)

    result = {
        "pairs": [
            {"source": s, "translation": t}
            for s, t in zip(cleaned, translations)
        ],
        "scores": {},
    }

    if references and all(r for r in refs) and len(translations) == len(refs):
        result["scores"] = metrics.score(translations, refs)
        for item, ref in zip(result["pairs"], refs):
            item["reference"] = ref

    if embed and translations:
        similarity = semantic_similarity(cleaned, translations)
        if similarity is not None:
            result["scores"]["mean_semantic_similarity"] = similarity

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", nargs="+", help="one or more sentences")
    source.add_argument("--corpus", type=Path, help="JSON array of {id, text}")
    source.add_argument("--hypotheses", type=Path,
                        help="score existing translations, one per line (no model needed)")

    parser.add_argument("--references", type=Path, help="gold translations, one per line")
    parser.add_argument("--reference-key", default=None,
                        help="key holding the gold translation in --corpus records")
    parser.add_argument("--limit", type=int, default=10,
                        help="records to take from --corpus (default: 10)")
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"source lang (default: {DEFAULT_SRC})")
    parser.add_argument("--tgt", default=DEFAULT_TGT, help=f"target lang (default: {DEFAULT_TGT})")
    parser.add_argument("--no-embed", action="store_true", help="skip sentence embeddings")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    # Scoring pre-translated output needs no model at all.
    if args.hypotheses:
        if not args.references:
            parser.error("--hypotheses requires --references")
        hyps, refs = read_lines(args.hypotheses), read_lines(args.references)
        if len(hyps) != len(refs):
            parser.error(f"{len(hyps)} hypotheses vs {len(refs)} references — must match")

        scores = metrics.score(hyps, refs)
        print(json.dumps(scores, indent=2) if args.json else metrics.format_scores(scores))
        return

    if args.text:
        sentences, references = args.text, None
    else:
        records = load_corpus(args.corpus, limit=args.limit)
        sentences = [r.get("text", "") for r in records]
        references = ([r.get(args.reference_key) for r in records]
                      if args.reference_key else None)

    if args.references:
        references = read_lines(args.references)

    result = evaluate(sentences, references=references, src_lang=args.src,
                      tgt_lang=args.tgt, embed=not args.no_embed)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    for item in result["pairs"]:
        print(f"  {item['source']}\n-> {item['translation']}")
        if "reference" in item:
            print(f" ref {item['reference']}")
        print()

    if result["scores"]:
        print(metrics.format_scores(result["scores"]))
    elif references is None:
        print("No references supplied — translations only, no quality score.")


if __name__ == "__main__":
    main()
