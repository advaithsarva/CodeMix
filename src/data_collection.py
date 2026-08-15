"""Build a corpus file from a real code-mixed dataset.

This file was empty. The audit in corpus.py shows the bundled data/ files are
shuffled tokens rather than language, so the useful thing for this module to
do is convert a genuine corpus into the {"id", "text"} shape the rest of the
project reads.

Nothing here downloads anything — you fetch the dataset yourself (they have
their own licences and access terms), then point this at the file:

    LinCE          https://ritual.uh.edu/lince/
    GLUECoS        https://github.com/microsoft/GLUECoS
    L3Cube         https://github.com/l3cube-pune/code-mixed-nlp

Usage:
    python src/data_collection.py raw.txt        -o data/Processed/lince.json
    python src/data_collection.py raw.csv        -o out.json --column tweet
    python src/data_collection.py raw.conll      -o out.json --format conll
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from preprocessing import clean_text


def from_lines(path: Path):
    """One sentence per line."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line.strip()


def from_csv(path: Path, column: str):
    """One column of a CSV/TSV."""
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if column not in (reader.fieldnames or []):
            raise SystemExit(
                f"column {column!r} not in {path.name}; found: {reader.fieldnames}"
            )
        for row in reader:
            text = (row.get(column) or "").strip()
            if text:
                yield text


def from_conll(path: Path):
    """CoNLL-style token-per-line files, blank line between sentences.

    This is the format LinCE and GLUECoS ship in: the first whitespace-
    separated field is the token, later fields are language-ID tags.
    """
    tokens: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                if tokens:
                    yield " ".join(tokens)
                    tokens = []
                continue
            if line.startswith("#"):
                continue
            tokens.append(line.split()[0])
    if tokens:
        yield " ".join(tokens)


READERS = {"lines": from_lines, "csv": from_csv, "conll": from_conll}


def build(sentences, *, clean: bool = True, min_tokens: int = 2) -> list[dict]:
    """Turn raw sentences into deduplicated {id, text} records."""
    seen: set[str] = set()
    records: list[dict] = []

    for sentence in sentences:
        text = clean_text(sentence) if clean else sentence.strip()
        if not text or len(text.split()) < min_tokens or text in seen:
            continue
        seen.add(text)
        records.append({"id": len(records), "text": text})

    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="raw dataset file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="JSON to write")
    parser.add_argument("--format", choices=sorted(READERS), default="lines")
    parser.add_argument("--column", default="text", help="CSV column (--format csv)")
    parser.add_argument("--min-tokens", type=int, default=2)
    parser.add_argument("--no-clean", action="store_true", help="keep text verbatim")
    args = parser.parse_args(argv)

    if not args.source.exists():
        raise SystemExit(f"no such file: {args.source}")

    if args.format == "csv":
        sentences = from_csv(args.source, args.column)
    else:
        sentences = READERS[args.format](args.source)

    records = build(sentences, clean=not args.no_clean, min_tokens=args.min_tokens)
    if not records:
        raise SystemExit("no usable sentences found — check --format and --column")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=1)

    print(f"wrote {len(records)} records to {args.output}")
    print("Audit it with:  python src/corpus.py", args.output)


if __name__ == "__main__":
    main()
