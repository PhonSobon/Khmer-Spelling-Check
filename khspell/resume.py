"""Resume an interrupted training (power cut, Kaggle time limit, network error).

A training script saves its full state (model, optimizer, scheduler, position, random
generators) to `<model file>.resume` while it trains, and deletes it when it finishes.
So an existing .resume file means "this training is not finished": main.py runs the
step again and the script continues where it stopped instead of starting over.
"""
import os
import random
from pathlib import Path

import torch


def state_path(out):
    return Path(f"{out}.resume")


def unfinished(out):
    return state_path(out).exists()


def save(out, **state):
    """Write atomically: a crash while saving never leaves a broken state file."""
    path = state_path(out)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({**state, "py_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp)
    os.replace(tmp, path)


def load(out, fresh=False):
    """The saved state, or None (no state, or --fresh). Restores the global random generators."""
    path = state_path(out)
    if fresh or not path.exists():
        return None
    st = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(st["py_rng"])
    torch.set_rng_state(st["torch_rng"])
    if st.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda_rng"])
    return st


def finish(out):
    """Training completed: remove the resume state."""
    state_path(out).unlink(missing_ok=True)
