import os
import torch

from model import SACB_Net


def main():
    inshape = (32, 32, 32)
    if not torch.cuda.is_available():
        raise RuntimeError("smoke_forward_test requires CUDA.")
    device = torch.device("cuda")
    force_cpu_fft = os.environ.get("SACB_FORCE_CPU_FFT", "0") == "1"

    model = SACB_Net(inshape=inshape, lp_ratio=0.15).to(device)
    model.set_force_cpu_fft(force_cpu_fft)
    model.eval()

    x = torch.randn(1, 1, *inshape, device=device)
    y = torch.randn(1, 1, *inshape, device=device)

    with torch.no_grad():
        x_warped, phi = model(x, y)

    print("device:", device)
    print("force_cpu_fft:", force_cpu_fft)
    print("x_warped shape:", tuple(x_warped.shape))
    print("phi shape:", tuple(phi.shape))


if __name__ == "__main__":
    main()
