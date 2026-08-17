#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.k3_vendor_verifier import gate_deepswe, load_contract


class K3VendorVerifierContractTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("benchmarks/k3_vendor_verifier_contract.json")
        self.contract = load_contract(self.path)

    def test_pinned_upstream_and_targets(self):
        self.assertEqual(
            self.contract["upstream"]["commit"],
            "3dad65a760a8867cda72f6dd8848d876a4e851b4",
        )
        self.assertEqual(
            self.contract["official_targets"],
            {
                "OCRBench": 0.89,
                "MMMU Pro Vision": 0.82,
                "BEAM (1M)": 0.31,
                "DeepSWE": 0.675,
            },
        )

    def test_k3_max_profile_is_exactly_pinned(self):
        self.assertEqual(
            self.contract["k3_max"],
            {
                "thinking": True,
                "keep": "all",
                "reasoning_effort": "max",
                "temperature": 1.0,
                "top_p": 0.95,
            },
        )
        self.assertEqual(
            self.contract["official_eval_parameters"]["MMMU Pro Vision"]["max_tokens"],
            98304,
        )
        self.assertEqual(
            self.contract["official_eval_parameters"]["BEAM (1M)"]["context"],
            1048576,
        )

    def test_bad_schema_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text(json.dumps({"schema": 999}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                load_contract(p)

    def test_deepswe_gate_accepts_target_and_rejects_regression(self):
        gate_deepswe(self.contract, 0.675)
        with self.assertRaises(SystemExit):
            gate_deepswe(self.contract, 0.6749)


if __name__ == "__main__":
    unittest.main()
