import unittest

try:
    import torch
except Exception:
    torch = None

IMPORT_ERR = None
FrequencyPartition = None
SACB = None

if torch is not None:
    try:
        from SACB1 import FrequencyPartition, SACB
    except Exception as e:  # pragma: no cover
        IMPORT_ERR = e


@unittest.skipIf(torch is None, 'torch is not installed in current environment')
@unittest.skipIf(FrequencyPartition is None or SACB is None, f'dependencies unavailable: {IMPORT_ERR}')
class TestSACBFFT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_frequency_partition_output_shape_and_binary(self):
        partition = FrequencyPartition(lp_ratio=0.2)
        x = torch.randn(2, 4, 8, 8, 8)

        idx = partition(x)

        self.assertEqual(tuple(idx.shape), (2, 8 * 8 * 8))
        uniq = set(idx.unique().tolist())
        self.assertTrue(uniq.issubset({0, 1}))

    def test_frequency_partition_ratio_update_clears_mask_cache(self):
        partition = FrequencyPartition(lp_ratio=0.1)
        x = torch.randn(1, 2, 6, 6, 6)

        _ = partition(x)
        self.assertGreater(len(partition._mask_cache), 0)

        partition.set_lp_ratio(0.3)
        self.assertEqual(len(partition._mask_cache), 0)

        _ = partition(x)
        self.assertGreater(len(partition._mask_cache), 0)

    def test_sacb_forward_shape_and_finite_for_bs1_and_bs2(self):
        for bs in (1, 2):
            module = SACB(in_ch=8, out_ch=8, ks=3, in_proj_n=1, mean_type='s', lp_ratio=0.15)
            module.eval()
            x = torch.randn(bs, 8, 8, 8, 8)

            with torch.no_grad():
                y = module(x)

            self.assertEqual(tuple(y.shape), tuple(x.shape))
            self.assertTrue(torch.isfinite(y).all().item())

    def test_frequency_partition_changes_with_lp_ratio(self):
        partition = FrequencyPartition(lp_ratio=0.01)
        d = h = w = 8
        zz, yy, xx = torch.meshgrid(
            torch.arange(d), torch.arange(h), torch.arange(w), indexing='ij'
        )

        # High-frequency checkerboard signal highlights lp_ratio effect.
        checker = ((zz + yy + xx) % 2).float() * 2.0 - 1.0
        x = checker.unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1, 1)

        idx_small = partition(x)
        partition.set_lp_ratio(1.8)
        idx_large = partition(x)

        self.assertFalse(torch.equal(idx_small, idx_large))
        diff_ratio = (idx_small != idx_large).float().mean().item()
        self.assertGreater(diff_ratio, 0.2)


if __name__ == '__main__':
    unittest.main()
