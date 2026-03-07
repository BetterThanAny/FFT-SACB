import unittest

import torch

import losses


class TestLossDevices(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _sample(self, device):
        x = torch.rand(1, 1, 8, 8, 8, device=device)
        y = torch.rand(1, 1, 8, 8, 8, device=device)
        return x, y

    def _assert_losses_on_device(self, device):
        x, y = self._sample(device)

        ncc = losses.NCC_vxm()
        mind = losses.MIND_loss()
        mi = losses.MutualInformation().to(device)
        lmi = losses.localMutualInformation().to(device)

        self.assertEqual(ncc(x, y).device, x.device)
        self.assertEqual(mind(x, y).device, x.device)
        self.assertEqual(mi(x, y).device, x.device)
        self.assertEqual(lmi(x, y).device, x.device)

    def test_cpu_losses_follow_input_device(self):
        self._assert_losses_on_device(torch.device('cpu'))

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA unavailable in current environment')
    def test_cuda_losses_follow_input_device(self):
        self._assert_losses_on_device(torch.device('cuda'))


if __name__ == '__main__':
    unittest.main()
