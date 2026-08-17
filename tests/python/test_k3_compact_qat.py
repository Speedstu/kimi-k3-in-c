import unittest

import numpy as np

from compact.q3 import pack_matrix, row_bytes, unpack_matrix

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from compact.qat import (
        Q3WeightQAT,
        estimated_matrix_bytes,
        fake_quant_q3,
        initial_group_scales,
        physical_bits_per_weight,
    )


@unittest.skipUnless(torch is not None, "Q3 QAT tests require PyTorch")
class K3CompactQATTests(unittest.TestCase):
    def test_initial_fake_quant_matches_reference_exporter(self):
        rng = np.random.default_rng(91)
        w_np = rng.normal(0.0, 0.4, size=(5, 259)).astype(np.float32)
        w = torch.from_numpy(w_np.copy())
        scales = initial_group_scales(w)
        fake = fake_quant_q3(w, scales).detach().cpu().numpy()
        deployed = unpack_matrix(pack_matrix(w_np), *w_np.shape)
        np.testing.assert_allclose(fake, deployed, rtol=0.0, atol=0.0)

    def test_learned_scale_and_weight_receive_finite_gradients(self):
        torch.manual_seed(4)
        weight = torch.nn.Parameter(torch.randn(4, 131, dtype=torch.float32) * 0.2)
        qat = Q3WeightQAT(weight.detach())
        x = torch.randn(3, 131)
        target = torch.randn(3, 4)
        out = x @ qat(weight).t()
        loss = torch.mean((out - target) ** 2)
        loss.backward()
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(qat.log_scale_raw.grad)
        self.assertTrue(torch.all(torch.isfinite(weight.grad)))
        self.assertTrue(torch.all(torch.isfinite(qat.log_scale_raw.grad)))
        self.assertGreater(float(weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(qat.log_scale_raw.grad.abs().sum()), 0.0)

    def test_physical_accounting_matches_reference_packer(self):
        for rows, cols in ((1, 128), (3, 131), (7, 259), (2, 1)):
            self.assertEqual(estimated_matrix_bytes(rows, cols), rows * row_bytes(cols))
        self.assertAlmostEqual(physical_bits_per_weight(), 3.125)


if __name__ == "__main__":
    unittest.main()
