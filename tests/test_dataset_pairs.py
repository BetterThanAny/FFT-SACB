import unittest

from dataset.datasets import LPBABrainDatasetS2S, LPBABrainInferDatasetS2S


def _identity(items):
    return items


class _PairAssertionsMixin:
    dataset_cls = None

    def make_dataset(self, n=4):
        paths = [f'subj_{i}.pkl' for i in range(n)]
        return self.dataset_cls(paths, transforms=_identity), paths

    def test_length_is_n_times_n_minus_1(self):
        ds, paths = self.make_dataset(n=5)
        self.assertEqual(len(ds), len(paths) * (len(paths) - 1))

    def test_no_self_pair(self):
        ds, _ = self.make_dataset(n=4)
        for idx in range(len(ds)):
            src, tgt = ds._pair_from_index(idx)
            self.assertNotEqual(src, tgt)

    def test_out_of_range_raises(self):
        ds, _ = self.make_dataset(n=3)
        with self.assertRaises(IndexError):
            ds._pair_from_index(-1)
        with self.assertRaises(IndexError):
            ds._pair_from_index(len(ds))

    def test_pair_coverage_matches_full_permutation_without_self(self):
        ds, paths = self.make_dataset(n=4)
        observed = set(ds._pair_from_index(i) for i in range(len(ds)))
        expected = {(paths[i], paths[j]) for i in range(len(paths)) for j in range(len(paths)) if i != j}
        self.assertEqual(observed, expected)


class TestLPBABrainDatasetS2SPairs(_PairAssertionsMixin, unittest.TestCase):
    dataset_cls = LPBABrainDatasetS2S


class TestLPBABrainInferDatasetS2SPairs(_PairAssertionsMixin, unittest.TestCase):
    dataset_cls = LPBABrainInferDatasetS2S


if __name__ == '__main__':
    unittest.main()
