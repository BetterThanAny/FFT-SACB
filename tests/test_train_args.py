import tempfile
import unittest
from pathlib import Path

import train


class TestTrainArgs(unittest.TestCase):
    def _parse(self, *extra_args):
        parser = train.build_parser()
        return parser.parse_args(list(extra_args))

    def test_validate_args_accepts_valid_input(self):
        with tempfile.TemporaryDirectory() as td:
            args = self._parse('--base-dir', td)
            train.validate_args(args)

    def test_validate_args_rejects_invalid_batch_size(self):
        with tempfile.TemporaryDirectory() as td:
            args = self._parse('--base-dir', td, '--batch-size', '0')
            with self.assertRaisesRegex(ValueError, '--batch-size'):
                train.validate_args(args)

    def test_validate_args_rejects_invalid_epoch_start(self):
        with tempfile.TemporaryDirectory() as td:
            args = self._parse('--base-dir', td, '--max-epoch', '2', '--epoch-start', '2')
            with self.assertRaisesRegex(ValueError, '--epoch-start'):
                train.validate_args(args)

    def test_validate_args_rejects_negative_val_batch_size(self):
        with tempfile.TemporaryDirectory() as td:
            args = self._parse('--base-dir', td, '--val-batch-size', '-1')
            with self.assertRaisesRegex(ValueError, '--val-batch-size'):
                train.validate_args(args)

    def test_resolve_resume_ckpt_prefers_explicit_path(self):
        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td)
            explicit = exp_dir / 'explicit_e9.pth.tar'
            latest = exp_dir / 'auto_e10.pth.tar'
            explicit.touch()
            latest.touch()

            resolved = train.resolve_resume_ckpt(exp_dir, str(explicit))
            self.assertEqual(resolved, explicit)

    def test_resolve_resume_ckpt_auto_latest(self):
        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td)
            (exp_dir / 'dsc0.6_e1.pth.tar').touch()
            (exp_dir / 'dsc0.7_e2.pth.tar').touch()
            newest = exp_dir / 'dsc0.8_e10.pth.tar'
            newest.touch()

            resolved = train.resolve_resume_ckpt(exp_dir)
            self.assertEqual(resolved, newest)

    def test_resolve_resume_ckpt_explicit_missing(self):
        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td)
            missing = exp_dir / 'missing.pth.tar'
            with self.assertRaisesRegex(FileNotFoundError, 'Resume checkpoint not found'):
                train.resolve_resume_ckpt(exp_dir, str(missing))

    def test_resolve_resume_ckpt_auto_empty(self):
        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td)
            with self.assertRaisesRegex(FileNotFoundError, 'No checkpoint found'):
                train.resolve_resume_ckpt(exp_dir)


if __name__ == '__main__':
    unittest.main()
