import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from compact.distill import (
        aggregate_teacher_router_probs,
        hidden_alignment_loss,
        logit_kl_loss,
        router_distill_loss,
        specialist_distill_loss,
        verified_improvement_loss,
    )


@unittest.skipUnless(torch is not None, "K3-Compact distillation tests require PyTorch")
class K3CompactDistillTests(unittest.TestCase):
    def test_router_cluster_aggregation_conserves_probability_mass(self):
        teacher = torch.tensor(
            [[0.05, 0.10, 0.15, 0.20, 0.10, 0.05, 0.25, 0.10]], dtype=torch.float32
        )
        mapping = torch.tensor([0, 0, 1, 1, 2, 2, 2, 1], dtype=torch.long)
        student = aggregate_teacher_router_probs(teacher, mapping, 3)
        expected = torch.tensor([[0.15, 0.45, 0.40]], dtype=torch.float32)
        torch.testing.assert_close(student, expected, rtol=0.0, atol=1e-7)
        torch.testing.assert_close(student.sum(-1), teacher.sum(-1), rtol=0.0, atol=1e-7)

    def test_identical_logits_and_hidden_have_near_zero_retention_loss(self):
        torch.manual_seed(3)
        logits = torch.randn(2, 5, 17)
        hidden = torch.randn(2, 5, 11)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
        self.assertLess(float(logit_kl_loss(logits, logits, mask=mask, temperature=2.0)), 1e-6)
        self.assertLess(float(hidden_alignment_loss(hidden, hidden, mask=mask)), 1e-6)

    def test_router_distill_is_near_zero_for_exact_aggregated_target(self):
        teacher_logits = torch.log(torch.tensor([[0.2, 0.3, 0.1, 0.4]], dtype=torch.float32))
        mapping = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        target = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        student_logits = torch.log(target)
        loss = router_distill_loss(student_logits, teacher_logits, mapping)
        self.assertLess(float(loss), 1e-6)

    def test_verified_loss_rewards_beating_teacher_margin(self):
        teacher_good = torch.tensor([-2.0])
        teacher_bad = torch.tensor([-3.0])  # teacher margin = +1
        weak = verified_improvement_loss(
            torch.tensor([-2.0]),
            torch.tensor([-2.7]),
            teacher_good,
            teacher_bad,
            margin_over_teacher=0.25,
        )
        strong = verified_improvement_loss(
            torch.tensor([-1.5]),
            torch.tensor([-3.0]),  # student margin = +1.5 > 1.25 target
            teacher_good,
            teacher_bad,
            margin_over_teacher=0.25,
        )
        self.assertLess(float(strong), float(weak))

    def test_full_objective_backpropagates_finite_gradients(self):
        torch.manual_seed(17)
        batch, seq, vocab, hidden = 2, 4, 13, 9
        teacher_experts, student_experts = 8, 3
        student_logits = torch.randn(batch, seq, vocab, requires_grad=True)
        teacher_logits = torch.randn(batch, seq, vocab)
        student_hidden = torch.randn(batch, seq, hidden, requires_grad=True)
        teacher_hidden = torch.randn(batch, seq, hidden)
        student_router = torch.randn(batch, seq, student_experts, requires_grad=True)
        teacher_router = torch.randn(batch, seq, teacher_experts)
        mapping = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
        chosen = torch.randn(batch, requires_grad=True)
        rejected = torch.randn(batch, requires_grad=True)
        teacher_chosen = torch.randn(batch)
        teacher_rejected = torch.randn(batch)

        losses = specialist_distill_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_hidden=student_hidden,
            teacher_hidden=teacher_hidden,
            student_router_logits=student_router,
            teacher_router_logits=teacher_router,
            teacher_to_student=mapping,
            token_mask=mask,
            student_chosen_logp=chosen,
            student_rejected_logp=rejected,
            teacher_chosen_logp=teacher_chosen,
            teacher_rejected_logp=teacher_rejected,
        )
        losses["total"].backward()
        for tensor in (student_logits, student_hidden, student_router, chosen, rejected):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.all(torch.isfinite(tensor.grad)))
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
