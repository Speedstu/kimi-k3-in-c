import unittest

import numpy as np

from compact.q3 import (
    GROUP,
    bf16_bits_to_f32,
    f32_to_bf16_bits,
    pack_codes,
    pack_matrix,
    pack_row,
    packed_code_bytes,
    row_bytes,
    unpack_codes,
    unpack_matrix,
    unpack_row,
)


class K3CompactQ3Tests(unittest.TestCase):
    def test_physical_budget_is_3_125_bits_for_full_groups(self):
        self.assertEqual(packed_code_bytes(GROUP), 48)
        self.assertEqual(row_bytes(GROUP), 50)
        self.assertEqual(row_bytes(2 * GROUP), 100)
        self.assertAlmostEqual(row_bytes(GROUP) * 8 / GROUP, 3.125)

    def test_signed_bitstream_all_alignments_and_tails(self):
        for n in list(range(1, 20)) + [63, 64, 65, 127, 128, 129]:
            q = np.asarray([((i * 5 + 2) % 8) - 4 for i in range(n)], dtype=np.int8)
            packed = pack_codes(q)
            self.assertEqual(len(packed), packed_code_bytes(n))
            np.testing.assert_array_equal(unpack_codes(packed, n), q)

    def test_bf16_scale_roundtrip_is_finite(self):
        for value in (0.001, 0.125, 0.5, 1.0, 3.25, 127.0):
            bits = f32_to_bf16_bits(value)
            got = bf16_bits_to_f32(bits)
            self.assertTrue(np.isfinite(got))
            self.assertGreater(got, 0.0)
            self.assertLess(abs(got - value) / value, 0.005)

    def test_reference_quantizer_roundtrips_shape_and_size(self):
        rng = np.random.default_rng(123)
        x = rng.normal(0.0, 0.35, size=259).astype(np.float32)
        packed = pack_row(x)
        y = unpack_row(packed, x.size)
        self.assertEqual(len(packed), row_bytes(x.size))
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(y)))
        # PTQ is only a format smoke; QAT is expected to recover quality. Still ensure
        # the reference encoder is sane rather than silently emitting all zeros.
        rel_rmse = float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(x**2)))
        self.assertLess(rel_rmse, 0.20)
        self.assertGreater(np.count_nonzero(y), x.size // 2)

    def test_matrix_roundtrip_has_no_padding_between_rows(self):
        rng = np.random.default_rng(4)
        w = rng.normal(size=(7, 131)).astype(np.float32)
        blob = pack_matrix(w)
        self.assertEqual(len(blob), 7 * row_bytes(131))
        y = unpack_matrix(blob, 7, 131)
        self.assertEqual(y.shape, w.shape)
        self.assertTrue(np.all(np.isfinite(y)))


if __name__ == "__main__":
    unittest.main()
