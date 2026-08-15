"""Test suite for CodeMix.

Run with:  python -m unittest discover -s tests -v

Uses the standard library only — no pytest, no network, and no model
downloads. The transformers-dependent code is exercised through its pure
helpers; the 600M-parameter model is never loaded.
"""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import corpus
import data_collection
import metrics
import translate
from preprocessing import clean_text, clean_records, script_of, tokenize

TELUGU_SENTENCE = "నేను office కి వెళ్తున్నాను"


class TestPreprocessing(unittest.TestCase):

    def test_telugu_script_survives_cleaning(self):
        # Regression: the original CleanText stripped [^\x00-\x7F], deleting
        # every Telugu character — the entire point of a code-mixed corpus.
        cleaned = clean_text(TELUGU_SENTENCE)
        self.assertIn("నేను", cleaned)
        self.assertIn("office", cleaned)

    def test_clean_text_takes_a_single_string(self):
        # Regression: CleanText(files_path, output_file, cleaned_output_file)
        # was called as df["text"].apply(CleanText) and raised TypeError.
        self.assertEqual(clean_text("hello world"), "hello world")

    def test_clean_text_tolerates_empty_and_non_string(self):
        for value in ("", None, 3.5, float("nan")):
            self.assertEqual(clean_text(value), "")

    def test_removes_urls_emails_and_handles(self):
        out = clean_text("mail me at a.b@x.com or https://x.com/y #tag @user ok")
        for fragment in ("a.b@x.com", "https://", "#tag", "@user"):
            self.assertNotIn(fragment, out)
        self.assertIn("ok", out)

    def test_digits_dropped_by_default_and_kept_on_request(self):
        self.assertNotIn("2024", clean_text("year 2024 here"))
        self.assertIn("2024", clean_text("year 2024 here", drop_digits=False))

    def test_whitespace_is_collapsed(self):
        self.assertEqual(clean_text("a   b \n\t c"), "a b c")

    def test_combining_vowel_signs_are_not_split(self):
        # NFC keeps Telugu vowel signs attached; without normalising, the
        # decomposed form tokenises into separate meaningless codepoints.
        decomposed = "కి"[0] + "ి"
        self.assertEqual(len(tokenize(clean_text(decomposed))), 1)

    def test_script_of(self):
        self.assertEqual(script_of("office"), "latin")
        self.assertEqual(script_of("నేను"), "telugu")
        self.assertEqual(script_of("goals్"), "mixed")
        self.assertEqual(script_of("123"), "other")

    def test_tokenize_strips_edge_punctuation(self):
        self.assertEqual(tokenize("hello, world!"), ["hello", "world"])

    def test_clean_records_drops_rows_that_clean_to_nothing(self):
        out = clean_records([
            {"id": 0, "text": "real text"},
            {"id": 1, "text": "###"},
            {"id": 2, "text": ""},
        ])
        self.assertEqual([r["id"] for r in out], [0])


class TestCorpusMetrics(unittest.TestCase):

    def test_code_mix_index_zero_for_single_language(self):
        self.assertEqual(corpus.code_mix_index(tokenize("this is all english")), 0.0)
        self.assertEqual(corpus.code_mix_index(tokenize("నేను వెళ్తున్నాను")), 0.0)

    def test_code_mix_index_rises_with_mixing(self):
        even = corpus.code_mix_index(["నేను", "office", "కి", "going"])
        lopsided = corpus.code_mix_index(["నేను", "నేను", "నేను", "office"])
        self.assertGreater(even, lopsided)
        self.assertAlmostEqual(even, 50.0, places=1)

    def test_code_mix_index_ignores_languageless_tokens(self):
        self.assertEqual(corpus.code_mix_index(["123", "!!!"]), 0.0)

    def test_adjacent_repeat_rate(self):
        self.assertEqual(corpus.adjacent_repeat_rate(["a", "b", "c"]), 0.0)
        self.assertAlmostEqual(corpus.adjacent_repeat_rate(["a", "a", "b"]), 0.5)
        self.assertEqual(corpus.adjacent_repeat_rate(["a"]), 0.0)

    def test_audit_flags_the_bundled_style_of_data_as_synthetic(self):
        # Shaped exactly like the real records in data/Processed: shuffled
        # tokens with immediate repeats and stray conjunctions.
        records = [
            {"id": 0, "text": "or సమయం or కాబట్టి and పూర్తి message message చాలా"},
            {"id": 1, "text": "but స్నేహితుడు సమయం message forecast ఫోన్"},
        ] * 50

        report = corpus.audit(records)
        self.assertTrue(corpus.looks_synthetic(report))
        self.assertIn("does not look like natural", corpus.format_report(report))

    def test_audit_accepts_natural_text(self):
        records = [
            {"id": 0, "text": "నేను office కి వెళ్తున్నాను and it is a long day"},
            {"id": 1, "text": "the meeting is at ఆఫీసు but I will be there on time"},
        ] * 50

        report = corpus.audit(records)
        self.assertFalse(corpus.looks_synthetic(report))

    def test_audit_counts_empty_records(self):
        report = corpus.audit([{"id": 0, "text": ""}, {"id": 1, "text": "hello there"}])
        self.assertEqual(report["empty_records"], 1)

    def test_load_corpus_respects_limit_and_rejects_non_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.json"
            good.write_text(json.dumps([{"id": i, "text": "x"} for i in range(10)]),
                            encoding="utf-8")
            self.assertEqual(len(corpus.load_corpus(good, limit=3)), 3)

            bad = Path(tmp) / "bad.json"
            bad.write_text('{"not": "an array"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                corpus.load_corpus(bad)


class TestTranslateHelpers(unittest.TestCase):
    """Pure helpers only — the model is never downloaded."""

    def test_correct_language_tag_for_telugu_script(self):
        # Regression: the original code asked for tel_Latn (romanised Telugu)
        # while the corpus is in Telugu script.
        self.assertEqual(translate.DEFAULT_SRC, "tel_Telu")
        self.assertEqual(translate.resolve_lang("tel_Telu"), "tel_Telu")

    def test_bare_iso_code_is_rejected_with_a_useful_message(self):
        with self.assertRaises(ValueError) as ctx:
            translate.resolve_lang("te")
        self.assertIn("tel_Telu", str(ctx.exception))

    def test_unknown_tag_is_rejected(self):
        with self.assertRaises(ValueError):
            translate.resolve_lang("xxx_Yyyy")

    def test_batched_splits_evenly_and_handles_remainder(self):
        self.assertEqual(list(translate.batched([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])
        self.assertEqual(list(translate.batched([], 3)), [])
        with self.assertRaises(ValueError):
            list(translate.batched([1], 0))

    def test_constructing_a_translator_downloads_nothing(self):
        # Importing and constructing must stay cheap; the model loads lazily.
        t = translate.Translator()
        self.assertIsNone(t._model)


class TestDataCollection(unittest.TestCase):

    def test_build_deduplicates_and_renumbers(self):
        records = data_collection.build(["hello world", "hello world", "another line"])
        self.assertEqual([r["id"] for r in records], [0, 1])
        self.assertEqual(records[0]["text"], "hello world")

    def test_build_drops_sentences_below_min_tokens(self):
        self.assertEqual(data_collection.build(["hi"], min_tokens=2), [])

    def test_build_preserves_telugu(self):
        records = data_collection.build([TELUGU_SENTENCE])
        self.assertIn("నేను", records[0]["text"])

    def test_conll_reader_groups_tokens_into_sentences(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.conll"
            path.write_text(
                "# comment\nనేను\tlang1\noffice\tlang2\n\nsecond\tlang2\nline\tlang2\n",
                encoding="utf-8",
            )
            self.assertEqual(list(data_collection.from_conll(path)),
                             ["నేను office", "second line"])

    def test_csv_reader_extracts_the_named_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            path.write_text("id,tweet\n1,hello there\n2,\n", encoding="utf-8")
            self.assertEqual(list(data_collection.from_csv(path, "tweet")), ["hello there"])

    def test_csv_reader_fails_loudly_on_a_missing_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            path.write_text("id,tweet\n1,hi\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                list(data_collection.from_csv(path, "nope"))


class TestMetrics(unittest.TestCase):
    """Reference-based translation scoring. Stdlib only, no sacrebleu."""

    HYP = "the cat sat on the mat"

    def test_identical_text_scores_100(self):
        self.assertAlmostEqual(metrics.bleu([self.HYP], [self.HYP]), 100.0, places=4)
        self.assertAlmostEqual(metrics.chrf([self.HYP], [self.HYP]), 100.0, places=4)

    def test_short_identical_sentences_still_score_100(self):
        # Regression: a 2-token sentence has no 3- or 4-grams. Averaging over
        # those empty orders scored a perfect translation as 0.0. BLEU now
        # uses the effective order, as sacrebleu does.
        short = "ఆఫీసుకి వెళ్తున్నాను"
        self.assertAlmostEqual(metrics.bleu([short], [short]), 100.0, places=4)
        self.assertAlmostEqual(metrics.bleu(["one two"], ["one two"]), 100.0, places=4)
        self.assertAlmostEqual(metrics.bleu(["solo"], ["solo"]), 100.0, places=4)

    def test_completely_different_text_scores_near_zero(self):
        self.assertLess(metrics.bleu(["aaa bbb ccc ddd"], ["xxx yyy zzz www"]), 1.0)
        self.assertLess(metrics.chrf(["aaa bbb ccc ddd"], ["xxx yyy zzz www"]), 10.0)

    def test_partial_match_lands_between(self):
        score = metrics.chrf(["the cat sat on the rug"], [self.HYP])
        self.assertGreater(score, 50.0)
        self.assertLess(score, 100.0)

    def test_bleu_applies_a_brevity_penalty(self):
        # A short hypothesis whose every n-gram is correct must still be
        # punished, or truncating output would look like perfect translation.
        full = metrics.bleu([self.HYP], [self.HYP])
        truncated = metrics.bleu(["the cat sat on"], [self.HYP])
        self.assertLess(truncated, full)

    def test_bleu_does_not_reward_repeated_ngrams(self):
        # Clipping: repeating a correct word must not inflate precision.
        spam = metrics.bleu(["the the the the the the"], [self.HYP])
        self.assertLess(spam, 60.0)

    def test_chrf_is_recall_weighted_by_default(self):
        # beta=2 weights recall twice as heavily, so dropping content should
        # cost more than adding it.
        dropped = metrics.chrf(["the cat"], [self.HYP])
        added = metrics.chrf([self.HYP + " today please"], [self.HYP])
        self.assertLess(dropped, added)

    def test_chrf_handles_telugu_script(self):
        hyp = "నేను ఆఫీసుకి వెళ్తున్నాను"
        self.assertAlmostEqual(metrics.chrf([hyp], [hyp]), 100.0, places=4)
        self.assertLess(metrics.chrf([hyp], ["పూర్తిగా వేరే వాక్యం"]), 40.0)

    def test_chrf_beats_bleu_on_inflection(self):
        # The reason chrF++ is the primary metric here: one inflected form
        # is a total miss for word-level BLEU but a near hit at char level.
        hyp, ref = ["ఆఫీసుకి వెళ్తున్నాను"], ["ఆఫీసుకు వెళ్తున్నాను"]
        self.assertGreater(metrics.chrf(hyp, ref), metrics.bleu(hyp, ref))

    def test_word_order_zero_gives_plain_chrf(self):
        hyp, ref = ["the cat sat"], ["the cat ran"]
        self.assertNotEqual(metrics.chrf(hyp, ref, word_order=0),
                            metrics.chrf(hyp, ref, word_order=2))

    def test_empty_input_scores_zero(self):
        self.assertEqual(metrics.bleu([], []), 0.0)
        self.assertEqual(metrics.chrf([], []), 0.0)
        self.assertEqual(metrics.chrf([""], [""]), 0.0)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.bleu(["a"], ["a", "b"])
        with self.assertRaises(ValueError):
            metrics.chrf(["a"], ["a", "b"])

    def test_ngrams(self):
        self.assertEqual(metrics.ngrams(["a", "b", "c"], 2),
                         Counter([("a", "b"), ("b", "c")]))
        self.assertEqual(metrics.ngrams(["a"], 2), Counter())
        self.assertEqual(metrics.ngrams(["a"], 0), Counter())

    def test_cosine_similarity(self):
        self.assertAlmostEqual(metrics.cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(metrics.cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(metrics.cosine_similarity([1, 0], [-1, 0]), -1.0)
        self.assertEqual(metrics.cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_score_reports_all_metrics(self):
        result = metrics.score([self.HYP], [self.HYP])
        self.assertEqual(result["sentences"], 1)
        for key in ("bleu", "chrf", "chrf2"):
            self.assertAlmostEqual(result[key], 100.0, places=1)

    def test_format_scores_is_readable(self):
        text = metrics.format_scores(metrics.score([self.HYP], [self.HYP]))
        self.assertIn("chrF++", text)
        self.assertIn("BLEU", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
