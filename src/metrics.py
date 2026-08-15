"""Translation quality metrics.

Pure standard library, so scoring and its tests run without sacrebleu,
transformers, or a network connection.

Two metrics, deliberately:

- **chrF++** is the primary score here. Character n-gram F-score handles
  morphologically rich, agglutinative languages far better than word-level
  BLEU, which is exactly the Telugu/Hindi situation — a single surface word
  carries inflection that BLEU counts as an outright miss.
- **BLEU** is reported too because it is what most papers quote, so results
  stay comparable even where it is the weaker measure.

Both are corpus-level, matching sacrebleu's definitions. They are not a
drop-in replacement for sacrebleu's tokenizer handling; for publication,
score with sacrebleu directly. These exist so the pipeline has an honest
number attached to it without adding a dependency.
"""

from __future__ import annotations

import math
from collections import Counter


def ngrams(tokens, n: int) -> Counter:
    """Count n-grams of length n."""
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _tokens(text: str) -> list[str]:
    return text.split()


def _chars(text: str) -> list[str]:
    """Characters with whitespace removed, per the chrF definition."""
    return list("".join(text.split()))


def _fscore(matched: float, hyp_total: float, ref_total: float, beta: float) -> float:
    if hyp_total == 0 or ref_total == 0:
        return 0.0
    precision = matched / hyp_total
    recall = matched / ref_total
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def bleu(hypotheses, references, max_order: int = 4) -> float:
    """Corpus BLEU, 0-100.

    Modified n-gram precision with clipping, geometric mean over orders 1..4,
    and the standard brevity penalty. Uses add-epsilon smoothing so a single
    missing higher-order match does not collapse the whole score to zero on
    short corpora.
    """
    hypotheses = list(hypotheses)
    references = list(references)
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")
    if not hypotheses:
        return 0.0

    matches = [0] * max_order
    totals = [0] * max_order
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        h_tokens, r_tokens = _tokens(hyp), _tokens(ref)
        hyp_len += len(h_tokens)
        ref_len += len(r_tokens)

        for order in range(1, max_order + 1):
            h_grams = ngrams(h_tokens, order)
            r_grams = ngrams(r_tokens, order)
            # Clip each hypothesis n-gram count by its count in the reference.
            matches[order - 1] += sum(min(c, r_grams[g]) for g, c in h_grams.items())
            totals[order - 1] += max(sum(h_grams.values()), 0)

    if hyp_len == 0:
        return 0.0

    # Effective order: average only over n-gram orders the data can actually
    # support. A 2-token sentence has no 3- or 4-grams, and counting those
    # empty orders would score even a perfect translation as 0.
    log_sum = 0.0
    used = 0
    for i in range(max_order):
        if totals[i] == 0:
            continue
        # Add-epsilon smoothing keeps a zero higher-order count from
        # annihilating the geometric mean.
        numerator = matches[i] if matches[i] > 0 else 1e-9
        log_sum += math.log(numerator / totals[i])
        used += 1

    if used == 0:
        return 0.0

    geometric_mean = math.exp(log_sum / used)
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100.0 * brevity * geometric_mean


def chrf(hypotheses, references, char_order: int = 6, word_order: int = 2,
         beta: float = 2.0) -> float:
    """Corpus chrF++ score, 0-100.

    char_order=6, word_order=2, beta=2 are the chrF++ defaults. Set
    word_order=0 for plain chrF. beta=2 weights recall twice as heavily as
    precision, which is the published default.
    """
    hypotheses = list(hypotheses)
    references = list(references)
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")
    if not hypotheses:
        return 0.0

    orders = char_order + word_order
    matched = [0] * orders
    hyp_total = [0] * orders
    ref_total = [0] * orders

    for hyp, ref in zip(hypotheses, references):
        h_chars, r_chars = _chars(hyp), _chars(ref)
        h_words, r_words = _tokens(hyp), _tokens(ref)

        for n in range(1, char_order + 1):
            _accumulate(ngrams(h_chars, n), ngrams(r_chars, n), n - 1,
                        matched, hyp_total, ref_total)

        for n in range(1, word_order + 1):
            _accumulate(ngrams(h_words, n), ngrams(r_words, n), char_order + n - 1,
                        matched, hyp_total, ref_total)

    # Average the per-order F-scores, skipping orders with no material.
    scores = [
        _fscore(matched[i], hyp_total[i], ref_total[i], beta)
        for i in range(orders)
        if hyp_total[i] or ref_total[i]
    ]
    return 100.0 * sum(scores) / len(scores) if scores else 0.0


def _accumulate(h_grams, r_grams, slot, matched, hyp_total, ref_total):
    matched[slot] += sum(min(c, r_grams[g]) for g, c in h_grams.items())
    hyp_total[slot] += sum(h_grams.values())
    ref_total[slot] += sum(r_grams.values())


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def score(hypotheses, references) -> dict:
    """All reference-based metrics at once."""
    return {
        "sentences": len(list(hypotheses)),
        "bleu": round(bleu(hypotheses, references), 2),
        "chrf2": round(chrf(hypotheses, references), 2),
        "chrf": round(chrf(hypotheses, references, word_order=0), 2),
    }


def format_scores(scores: dict) -> str:
    lines = ["Translation quality", "=" * 40]
    labels = {
        "sentences": "sentences scored",
        "chrf2":     "chrF++ (primary)",
        "chrf":      "chrF",
        "bleu":      "BLEU",
    }
    for key in ("sentences", "chrf2", "chrf", "bleu"):
        if key in scores:
            lines.append(f"{labels[key]:<24} {scores[key]}")

    if "mean_semantic_similarity" in scores:
        lines.append(f"{'mean cosine similarity':<24} {scores['mean_semantic_similarity']}")

    lines.append("=" * 40)
    return "\n".join(lines)
