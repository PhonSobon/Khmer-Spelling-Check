"""Check that PyTorch sees the GPU(s) before a long training run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from khspell import config

if not torch.cuda.is_available():
    sys.exit(f"CUDA not available (torch {torch.__version__}). Install the CUDA build of PyTorch from pytorch.org.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"torch {torch.__version__} | GPU {i}: {p.name} | {p.total_memory / 2**30:.0f} GB")
dtype = torch.bfloat16 if config.bf16_ok() else torch.float32
print(f"training precision: {'bf16' if dtype == torch.bfloat16 else 'fp32 (no native bf16 on this GPU)'}")
x = torch.randn(4096, 4096, device="cuda", dtype=dtype)
print("matmul OK:", float((x @ x).float().abs().mean()) > 0)
