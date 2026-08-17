import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from compact.distill import logit_kl_loss
    from compact.teacher_trace import (
        bucketed_topk_kl_loss,
        make_topk_trace,
        trace_physical_bytes_per_token,
    )


@unittest.skipUnless(torch is not None, "teacher trace tests require PyTorch")
class K3CompactTeacherTraceTests(unittest.TestCase):
    def test_trace_preserves_probability_mass(self):
        torch.manual_seed(8)
        teacher = torch.randn(2, 3, 37)
        trace = make_topk_trace(teacher, 7)
        mass = torch.exp(trace.log_probs).sum(-1) + torch.exp(trace.log_tail_mass)
        torch.testing.assert_close(mass, torch.ones_like(mass), rtol=1e-6, atol=1e-6)

    def test_identical_student_has_near_zero_bucketed_kl(self):
        torch.manual_seed(9)
        logits = torch.randn(4, 31)
        trace = make_topk_trace(logits, 8)
        loss = bucketed_topk_kl_loss(logits, trace)
        self.assertLess(abs(float(loss)), 2e-6)

    def test_vocab_minus_one_trace_matches_full_kl(self):
        torch.manual_seed(10)
        teacher = torch.randn(2, 4, 13)
        student = torch.randn(2, 4, 13)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
        trace = make_topk_trace(teacher, 12)
        bucket = bucketed_topk_kl_loss(student, trace, mask=mask)
        full = logit_kl_loss(student, teacher, mask=mask, temperature=1.0)
        torch.testing.assert_close(bucket, full, rtol=2e-5, atol=2e-6)

    def test_bucketed_loss_backpropagates(self):
        torch.manual_seed(11)
        teacher = torch.randn(3, 19)
        student = torch.randn(3, 19, requires_grad=True)
        trace = make_topk_trace(teacher, 5)
        loss = bucketed_topk_kl_loss(student, trace)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.all(torch.isfinite(student.grad)))
        self.assertGreater(float(student.grad.abs().sum()), 0.0)

    def test_trace_storage_is_small(self):
        # int32 token id + fp16 logprob, plus one fp16 tail bucket.
        self.assertEqual(trace_physical_bytes_per_token(128), 770)
        self.assertLess(trace_physical_bytes_per_token(128), 1024)


if __name__ == "__main__":
    unittest.main()
