import json
import unittest
from pathlib import Path

import numpy as np

from compact.expert_cluster import cluster_experts
from compact.plan import estimate
from compact.score_gate import evaluate


ROOT = Path(__file__).resolve().parents[2]


class K3CompactTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((ROOT / "compact/k3_compact.json").read_text(encoding="utf-8"))
        self.baselines = json.loads(
            (ROOT / "compact/k3_max_target_baselines.json").read_text(encoding="utf-8")
        )

    def test_architecture_fits_under_100gb(self):
        r = estimate(self.spec)
        self.assertAlmostEqual(r["total_params_b"], 202.604763904, places=6)
        self.assertAlmostEqual(r["active_params_b"], 68.898740992, places=6)
        self.assertTrue(r["fits_checkpoint_budget"])
        self.assertLess(r["estimated_checkpoint_gb"], 90.0)
        self.assertGreater(r["parameter_reduction_x"], 10.0)
        self.assertGreater(r["active_reduction_x"], 1.4)

    def test_behavior_clustering_is_complete_and_deterministic(self):
        rng = np.random.default_rng(7)
        centers = np.eye(3, 6)
        sketches = np.vstack(
            [centers[g] + rng.normal(0.0, 0.01, 6) for g in range(3) for _ in range(4)]
        )
        usage = np.arange(1, 13, dtype=np.float64)
        a = cluster_experts(sketches, usage, 3)
        b = cluster_experts(sketches, usage, 3)
        self.assertEqual(a["mapping"], b["mapping"])
        self.assertEqual(len(a["mapping"]), 12)
        self.assertEqual(len(a["clusters"]), 3)
        self.assertEqual(sorted(len(c["teacher_members"]) for c in a["clusters"]), [4, 4, 4])
        medoids = {c["teacher_medoid"] for c in a["clusters"]}
        self.assertEqual(len(medoids), 3)

    def test_score_gate_passes_only_on_real_head_to_head_wins(self):
        results = {
            "teacher": {"cyber": {"cybench": 40.0}},
            "student": {
                "code": {
                    "deepswe": 68.0,
                    "programbench": 78.0,
                    "terminal_bench_2_1": 88.5,
                    "kimi_code_bench_2_0": 73.1,
                },
                "agentic": {
                    "browsecomp": 91.3,
                    "deepsearchqa_f1": 95.1,
                    "researchrubrics": 76.3,
                },
                "cyber": {"cybench": 40.5},
            },
            "checkpoint_gb": 89.0,
            "general_retention_relative": 0.98,
        }
        report = evaluate(self.baselines, results)
        self.assertTrue(report["pass"])
        self.assertIn("K3-Compact > K3 Max", report["claim"])

    def test_score_gate_fails_closed_without_teacher_cyber_measurement(self):
        results = {
            "student": {
                "code": {m: 100.0 for m in ("deepswe", "programbench", "terminal_bench_2_1", "kimi_code_bench_2_0")},
                "agentic": {m: 100.0 for m in ("browsecomp", "deepsearchqa_f1", "researchrubrics")},
                "cyber": {"cybench": 100.0},
            },
            "checkpoint_gb": 80.0,
            "general_retention_relative": 1.0,
        }
        report = evaluate(self.baselines, results)
        self.assertFalse(report["pass"])
        self.assertEqual(report["claim"], "NOT PROVEN")
        self.assertEqual(report["cyber"]["missing"], ["cybench"])


if __name__ == "__main__":
    unittest.main()
