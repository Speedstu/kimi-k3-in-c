import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from compact.train_smoke import run_smoke


@unittest.skipUnless(torch is not None, "K3-Compact optimizer smoke requires PyTorch")
class K3CompactTrainSmokeTests(unittest.TestCase):
    def test_optimizer_recovers_retention_and_beats_teacher_margin(self):
        report = run_smoke(steps=300, lr=0.08)
        self.assertTrue(report["pass"], report)
        before = report["before"]
        after = report["after"]
        self.assertLess(after["total"], before["total"])
        self.assertGreater(after["student_margin"], after["teacher_margin"] + 0.25)
        self.assertLess(after["logits"], 1e-3)
        self.assertLess(after["router"], 1e-3)


if __name__ == "__main__":
    unittest.main()
