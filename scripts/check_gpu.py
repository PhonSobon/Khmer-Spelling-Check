"""Check that PyTorch sees the GPU before a long training run."""
import sys

import torch

if not torch.cuda.is_available():
    sys.exit(f"CUDA not available (torch {torch.__version__}). Install the CUDA build of PyTorch from pytorch.org.")
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | {p.name} | {p.total_memory / 2**30:.0f} GB | bf16: {torch.cuda.is_bf16_supported()}")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul OK:", float((x @ x).float().abs().mean()) > 0)
