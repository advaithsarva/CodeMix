# CodeMix

Telugu-English **code-mixed** text processing: clean it, measure it, translate
it to Hindi with NLLB-200, and embed it with a multilingual sentence
transformer.

Code-mixing is when both languages appear inside a single utterance:

```
నేను office కి వెళ్తున్నాను
"I am going to the office"
```

## Install

```bash
pip install -r requirements.txt
```

`preprocessing.py`, `corpus.py`, `metrics.py` and `data_collection.py` use the
standard library only — you can audit corpora, prepare datasets and score
translations without installing anything. The dependencies are needed only for
translation and embeddings.

Verified working with `transformers` 5.12.1 / `huggingface_hub` 1.27.0 /
`sentence-transformers` 5.6.0 / `torch` 2.7.1:

```
$ python src/app.py translate --text "నేను office కి వెళ్తున్నాను" --no-embed
  నేను office కి వెళ్తున్నాను
-> मैं ऑफिस जा रहा हूँ
```

## Usage

```bash
# Is a corpus actually real language?
python src/app.py audit data/Processed/code1.json

# Translate code-mixed Telugu-English to Hindi
python src/app.py translate --text "మన goals ద్వారా we can do wonders"
python src/app.py translate --corpus data/Processed/code1.json --limit 20

# Translate and score against gold references
python src/app.py translate --corpus pairs.json --reference-key hindi

# Score translations you already have (no model needed)
python src/app.py translate --hypotheses out.txt --references gold.txt

# Convert a real dataset into this project's JSON format
python src/app.py build raw.conll -o data/Processed/lince.json --format conll
```

## Evaluation

Reference-based scoring lives in `src/metrics.py` — standard library only, so
it runs without `sacrebleu` or a network connection.

| Metric | Role |
|---|---|
| **chrF++** | Primary. Character n-gram F-score (`char_order=6, word_order=2, beta=2`) |
| chrF | Same without word n-grams |
| BLEU | Reported for comparability with published results |
| mean cosine similarity | Reference-free sanity signal from the multilingual embedder |

**chrF++ is primary because BLEU is a poor fit for Telugu and Hindi.** Both
are morphologically rich, so a single inflected word that differs from the
reference is a total miss at the word level but a near hit at the character
level. Scoring `ఆఫీసుకి వెళ్తున్నాను` against `ఆఫీసుకు వెళ్తున్నాను` — one vowel
apart — makes the gap concrete:

```
chrF++   59.69
chrF     71.65
BLEU      0.00
```

BLEU uses the effective n-gram order, so short sentences are not scored as
zero simply for having no 4-grams. Cosine similarity is reported alongside
chrF++ and never instead of it — it cannot distinguish a fluent wrong answer
from a correct one.

For publication, score with `sacrebleu` directly; these implementations exist
so the pipeline has an honest number without adding a dependency.

## Tests

```bash
python -m unittest discover -s tests
```

45 tests, standard library `unittest`. No network access, and the 600M
parameter model is never downloaded — `translate.py` loads models lazily so
its language-tag validation and batching stay testable on their own.

## Status of the bundled data

**The corpus in `data/` is not usable for research.** It is 1,000,000 records
that look like code-mixed sentences but are randomly sampled tokens. Running
the audit on it:

```
vocabulary_size                  36
type_token_ratio                 0.0012
mean_adjacent_repeat_rate        0.0276
function_word_share_of_english   0.2776
most_common_tokens               forecast(895), అయితే(887), schedule(879), ...
```

A 36-word vocabulary spread over 30k tokens, with every frequent word
appearing a near-identical number of times, is the signature of uniform random
sampling. Natural language is Zipfian — a few words dominate and frequency
falls off sharply. Real code-mixed text also switches language at grammatical
boundaries, whereas these records alternate arbitrarily.

Any BLEU, chrF, or accuracy figure computed on this data would be meaningless.

**Next step:** replace it with an established corpus and convert with
`src/data_collection.py`:

| Corpus | Link |
|---|---|
| LinCE | https://ritual.uh.edu/lince/ |
| GLUECoS | https://github.com/microsoft/GLUECoS |
| L3Cube code-mixed NLP | https://github.com/l3cube-pune/code-mixed-nlp |

## Layout

```
src/preprocessing.py     clean_text(), script-aware tokenising
src/corpus.py            corpus loading, CMI, quality audit
src/metrics.py           chrF++, chrF, BLEU, cosine similarity
src/translate.py         NLLB translation + sentence embeddings
src/evaluation.py        translate and score
src/data_collection.py   convert a real dataset into corpus JSON
src/app.py               CLI entry point
tests/test_codemix.py    45 tests
data/                    corpora (gitignored, ~1GB)
```

## Notes on correctness

Several bugs in the earlier version are now fixed and covered by regression
tests:

- **Cleaning deleted all Telugu.** The old `CleanText` stripped
  `[^\x00-\x7F]`, removing every non-ASCII character — which in a
  Telugu-English corpus means the Telugu half of the data.
- **Cleaning crashed.** `CleanText` took three file paths but was invoked as
  `df["text"].apply(CleanText)`, which passes a single string.
- **English-only normalisation on Telugu.** The Porter stemmer and WordNet
  lemmatizer were applied to every token regardless of script.
- **NLLB was given the wrong language, in the wrong way.** The source language
  was passed as a `>>tel_Latn<< ` text prefix. That is the Marian / OPUS-MT
  convention; NLLB ignores it and translates the tag as literal text. NLLB
  takes the source language from `tokenizer.src_lang`.
- **Wrong language code.** `tel_Latn` is romanised Telugu; this corpus is in
  Telugu script, which is `tel_Telu`.
