import unittest

from compact.data_contract import TrainingRecord, normalized_text_hash, validate_records


def rec(sample_id: str, prompt: str, *, source: str = "synthetic/code/1") -> TrainingRecord:
    return TrainingRecord(
        sample_id=sample_id,
        domain="code",
        split="train",
        source_id=source,
        prompt_sha256=normalized_text_hash(prompt),
        chosen_sha256=normalized_text_hash(prompt + " chosen"),
        rejected_sha256=normalized_text_hash(prompt + " rejected"),
        verifier_id="unit-tests:v1",
        verifier_passed=True,
    )


class K3CompactDataContractTests(unittest.TestCase):
    def test_valid_verified_records_pass(self):
        records = [rec("a", "fix parser"), rec("b", "optimize loop")]
        counts = validate_records(records)
        self.assertEqual(counts["total"], 2)
        self.assertEqual(counts["code"], 2)

    def test_eval_hash_collision_fails_closed(self):
        r = rec("a", "held out exact prompt")
        with self.assertRaisesRegex(ValueError, "collides with held-out"):
            validate_records([r], denied_hashes={r.prompt_sha256})

    def test_eval_source_prefix_fails_closed(self):
        r = rec("a", "prompt", source="eval/deepswe/test/item-12")
        with self.assertRaisesRegex(ValueError, "held-out denylist"):
            validate_records([r], denied_source_prefixes=("eval/",))

    def test_duplicate_prompt_fails(self):
        a = rec("a", "same prompt")
        b = rec("b", "same prompt")
        with self.assertRaisesRegex(ValueError, "duplicate prompt"):
            validate_records([a, b])

    def test_unverified_pair_fails(self):
        r = rec("a", "prompt")
        r = TrainingRecord(**{**r.__dict__, "verifier_passed": False})
        with self.assertRaisesRegex(ValueError, "lacks a passing verifier"):
            validate_records([r])

    def test_test_split_is_never_trainable(self):
        r = rec("a", "prompt")
        r = TrainingRecord(**{**r.__dict__, "split": "test"})
        with self.assertRaisesRegex(ValueError, "held-out split"):
            validate_records([r])

    def test_line_ending_normalization_only(self):
        self.assertEqual(normalized_text_hash("a\r\nb"), normalized_text_hash("a\nb"))
        self.assertNotEqual(normalized_text_hash("A\nb"), normalized_text_hash("a\nb"))


if __name__ == "__main__":
    unittest.main()
