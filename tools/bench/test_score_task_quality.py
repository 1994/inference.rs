"""Task quality must follow raw answers and complete aligned inputs, not recorded totals."""

import copy
import importlib.util
import math
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "task_quality", Path(__file__).with_name("score-task-quality.py")
)
quality = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(quality)


def fixtures(count=256):
    samples = [
        {
            "id": str(i),
            "row_id": i,
            "variant": "long" if i % 2 else "short",
            "tokens": [100, i],
            "reference_answer": "Compute one plus one.\n#### 2",
        }
        for i in range(count)
    ]
    suite = {
        "profile_id": "test-quality",
        "samples": samples,
        "acceptance": {"noninferiority_margin_accuracy": 0.02, "confidence": 0.95},
    }
    report = {
        "completed": True,
        "profile_id": suite["profile_id"],
        "suite_sha256": "fixture-sha",
        "model": {"files": {"weights": "fixture-weights"}},
        "hardware": {"devices": ["fixture-device"]},
        "numeric_contract": {"state": "F32"},
        "results": [
            {
                "id": row["id"],
                "row_id": row["row_id"],
                "variant": row["variant"],
                "input_token_ids": row["tokens"],
                "text": "#### 2",
                "correct": True,
                "finish_reason": "Eos",
                "truncated": False,
                "prefix_tokens_reused": 0,
            }
            for row in samples
        ],
    }
    return suite, report, copy.deepcopy(report)


class AnswerTests(unittest.TestCase):
    def test_decimal_currency_percentage_and_sign(self):
        for text, expected in [("#### 1,250$", "1250"), ("#### -2.50", "-2.5"), ("#### 70%", "70")]:
            self.assertEqual(quality.numeric_answer(text), quality.Decimal(expected))

    def test_prose_numbers_are_not_final_answers(self):
        for text in [
            "The answer is 12",
            "#### NaN",
            "#### infinity",
            "#### 2e3",
            "#### 12 dollars",
        ]:
            self.assertIsNone(quality.numeric_answer(text))

    def test_long_integers_do_not_round_to_the_decimal_context(self):
        a = quality.numeric_answer("#### 1234567890123456789012345678901")
        b = quality.numeric_answer("#### 1234567890123456789012345678902")
        self.assertNotEqual(a, b)


class ConfidenceTests(unittest.TestCase):
    def test_zero_failures_matches_closed_form(self):
        self.assertAlmostEqual(quality.regression_upper_bound(0, 256, 0.95), 1 - 0.05 ** (1 / 256))

    def test_one_failure_inverts_known_binomial_probability(self):
        bound = quality.regression_upper_bound(1, 256, 0.95)
        probability = (1 - bound) ** 256 + 256 * bound * (1 - bound) ** 255
        self.assertAlmostEqual(probability, 0.05, places=11)
        self.assertLess(bound, 0.02)

    def test_two_failures_do_not_prove_two_percent_margin(self):
        self.assertGreater(quality.regression_upper_bound(2, 256, 0.95), 0.02)

    def test_large_counts_and_all_failures_are_finite(self):
        expected = sum(math.comb(2000, j) for j in range(801)) / (1 << 2000)
        self.assertAlmostEqual(
            math.exp(quality.log_binomial_cdf(800, 2000, 0.5)), expected, delta=1e-30
        )
        self.assertEqual(quality.regression_upper_bound(256, 256, 0.95), 1)

    def test_bad_counts_and_confidence_are_rejected(self):
        for k, n, c in [(0, 0, 0.95), (-1, 2, 0.95), (3, 2, 0.95), (0, 2, 1), (0, 2, math.nan)]:
            with self.assertRaises(ValueError):
                quality.regression_upper_bound(k, n, c)


class ReportTests(unittest.TestCase):
    def compare(self, suite, baseline, candidate):
        return quality.compare(suite, baseline, candidate, "fixture-sha")

    def test_identical_successes_pass_without_a_performance_claim(self):
        result = self.compare(*fixtures())
        self.assertTrue(result["quality_accepted"])
        self.assertFalse(result["performance_accepted"])

    def test_partial_duplicate_and_reordered_reports_fail(self):
        for edit in [
            lambda r: r.update(completed=False),
            lambda r: r["results"].pop(),
            lambda r: r["results"].reverse(),
            lambda r: r["results"].append(r["results"][0]),
        ]:
            suite, baseline, candidate = fixtures()
            edit(candidate)
            with self.assertRaises(ValueError):
                self.compare(suite, baseline, candidate)

    def test_identity_input_and_reuse_mismatches_fail(self):
        for edit in [
            lambda r: r.update(suite_sha256="other"),
            lambda r: r.update(profile_id="other"),
            lambda r: r.update(model={}),
            lambda r: r.update(hardware={}),
            lambda r: r.update(numeric_contract={"state": "BF16"}),
            lambda r: r["results"][0].update(input_token_ids=[2]),
            lambda r: r["results"][0].update(prefix_tokens_reused=1),
            lambda r: r["results"][0].update(finish_reason="Cancelled"),
            lambda r: r["results"][0].update(truncated=True),
            lambda r: r["results"][0].update(finished={"measurement": {"successful": False}}),
        ]:
            suite, baseline, candidate = fixtures()
            edit(candidate)
            with self.assertRaises(ValueError):
                self.compare(suite, baseline, candidate)

    def test_tampered_recorded_score_is_detected(self):
        suite, baseline, candidate = fixtures()
        candidate["results"][0]["text"] = "#### 3"
        with self.assertRaises(ValueError):
            self.compare(suite, baseline, candidate)

    def test_capped_missing_answer_remains_in_the_denominator(self):
        suite, baseline, candidate = fixtures()
        candidate["results"][0].update(
            text="Still calculating 2", correct=False, finish_reason="Length"
        )
        result = self.compare(suite, baseline, candidate)
        self.assertEqual(result["candidate"]["requests"], 256)
        self.assertEqual(result["candidate"]["length_limited"], 1)
        self.assertEqual(result["candidate"]["format_missing"], 1)
        self.assertEqual(result["regressions"], ["0"])

    def test_gains_do_not_hide_two_new_losses(self):
        suite, baseline, candidate = fixtures()
        for i in range(20):
            baseline["results"][i].update(text="#### 3", correct=False)
        for i in [20, 21]:
            candidate["results"][i].update(text="#### 3", correct=False)
        result = self.compare(suite, baseline, candidate)
        self.assertGreater(result["accuracy_change"], 0)
        self.assertFalse(result["quality_accepted"])


if __name__ == "__main__":
    unittest.main()
