"""Hugging Face access. The token is read from .env (HF_TOKEN) and never printed.

Big text datasets are handled as plain files: `download()` fetches only the data files
of one split (once, cached), and `iter_local_text()` reads them directly with pyarrow,
so a script can stop after the first N rows instead of converting a 1 GB file first.
"""
import os
from pathlib import Path

from . import config  # noqa: F401  (loads .env, silences the Windows symlink warning)

_OTHER_SPLITS = ("test", "validation", "valid", "dev")


def token():
    return os.environ.get("HF_TOKEN")


def load(ds_id, split="train", streaming=False):
    """Small datasets (dictionary, evaluation sets) through the `datasets` library."""
    from datasets import load_dataset

    return load_dataset(ds_id, split=split, token=token(), streaming=streaming)


def download(ds_id, split="train"):
    """Download the parquet/txt files of one split (cached after the first time). Returns local paths."""
    from huggingface_hub import HfApi, hf_hub_download

    files = [s.rfilename for s in HfApi(token=token()).dataset_info(ds_id).siblings]
    data = sorted(f for f in files if f.endswith((".parquet", ".txt")))
    chosen = [f for f in data if split in Path(f).name]
    if not chosen:   # e.g. data/segmented.txt: a single file without a split name
        chosen = [f for f in data if not any(s in Path(f).name for s in _OTHER_SPLITS)]
    if not chosen:
        raise FileNotFoundError(f"no parquet/txt files for split '{split}' in {ds_id}")
    return [hf_hub_download(ds_id, f, repo_type="dataset", token=token()) for f in chosen]


def iter_local_text(paths, column, batch_rows=2000):
    """Yield non-empty texts from downloaded parquet (column) or txt (one text per line) files."""
    import pyarrow.parquet as pq

    for p in paths:
        if str(p).endswith(".parquet"):
            for batch in pq.ParquetFile(p).iter_batches(batch_size=batch_rows, columns=[column]):
                for text in batch.column(0).to_pylist():
                    if text:
                        yield text
        else:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line


def iter_text(ds_id, column, split="train"):
    """Download (once) and read one split of a dataset: fast, and stops as early as the caller wants."""
    yield from iter_local_text(download(ds_id, split), column)
