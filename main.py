"""Train the Khmer spell checker (find misspelled words + top-k corrections) with one command.

    python main.py --quick      # test the whole process first      (~40 min on RTX 3090)
    python main.py              # full training                     (~4-5 h on RTX 3090)
    python main.py --serve      # after training: start the API at http://localhost:8000/docs

Other options:  --no-push (don't upload)  --retrain (train all models again)
                --rebuild-data (prepare the data again)

Steps. Finished steps are skipped when you run the command again, and an interrupted
ByT5 training continues from its last checkpoint, so after any error: fix it and rerun.
    1. check GPU (+ Hugging Face upload access)
    2. prepare data: download datasets, keep clean Khmer sentences   scripts/prepare_data.py
    3. lexicon: dictionary + frequent words of the data               scripts/build_lexicon.py
    4. G2P pronunciation model (same-sound spellings)                 scripts/train_g2p.py, build_pron.py
    5. character tagger (cut words + find errors)                     scripts/train_tagger.py
    6. ByT5 model (context-aware corrections)                         scripts/train_topk.py
    7. tune the top-k ranking + evaluate                              scripts/eval_topk.py

Everything is written to outputs/ (the log with all results: outputs/main.log).
The full run also uploads the models to the private Hugging Face repo
phonsobon/khmer-topk-speller (HF_TOKEN with write permission in .env).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from khspell import config  # noqa: E402  (also loads .env)

DEFAULT_HUB_REPO = "phonsobon/khmer-topk-speller"
DATA_VERSION_FILE = config.DATA_DIR / "VERSION.txt"


def run(name, script, *args):
    cmd = [sys.executable, "-u", str(ROOT / script), *map(str, args)]
    print(f"\n{'=' * 70}\n>>> {name}\n    python {' '.join(cmd[2:])}\n{'=' * 70}", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    start = time.time()
    with open(config.LOG, "a", encoding="utf-8") as log:
        log.write(f"\n>>> {name}: {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
    minutes = (time.time() - start) / 60
    if code != 0:
        print(f"\n!!! {name} FAILED (exit {code}) after {minutes:.1f} min. Details: {config.LOG}")
        print("    Fix the problem and run the same command again: finished steps are skipped.")
        sys.exit(code)
    print(f"--- {name}: done in {minutes:.1f} min", flush=True)


def skip(name, reason):
    print(f"\n>>> {name}: skipped ({reason})", flush=True)


def check_hub(repo):
    """Fail in seconds (not after hours of training) if the token cannot upload to `repo`."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("!!! HF_TOKEN missing: add HF_TOKEN=hf_... to .env, or run with --no-push")
    try:
        api = HfApi(token=token)
        user = api.whoami()["name"]
        api.create_repo(repo, private=True, exist_ok=True)
    except Exception as e:
        sys.exit(f"!!! Cannot upload to {repo}: {e}\n    The token needs WRITE permission, or use --no-push")
    print(f"    Hugging Face: logged in as {user}, models go to https://huggingface.co/{repo} (private)")


def upload_extras(repo, files):
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for f in files:
        if Path(f).exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f"speller/{Path(f).name}", repo_id=repo)
            print(f"    uploaded speller/{Path(f).name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="10%% of the data, short training: test everything")
    ap.add_argument("--retrain", action="store_true", help="train all models again")
    ap.add_argument("--rebuild-data", action="store_true", help="prepare the data (and lexicon) again")
    ap.add_argument("--steps", type=int, default=None, help="ByT5 training steps (default 30000, quick 3000)")
    ap.add_argument("--model", default="google/byt5-small", help="base model, e.g. google/byt5-base")
    ap.add_argument("--push-to-hub", default=None, help=f"Hugging Face repo (default {DEFAULT_HUB_REPO})")
    ap.add_argument("--no-push", action="store_true", help="do not upload anything")
    ap.add_argument("--serve", action="store_true", help="start the API at the end")
    ap.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)   # automatic; kept for old commands
    args = ap.parse_args()

    config.OUT.mkdir(parents=True, exist_ok=True)
    q = args.quick
    scale = "0.1" if q else "1.0"
    steps = args.steps or (3000 if q else 30000)
    topk_dir = config.OUT / ("topk_model_quick" if q else "topk_model")     # quick models never overwrite
    tagger_pt = config.OUT / ("tagger_quick.pt" if q else "tagger.pt")     # the real ones
    eval_done = config.OUT / ("EVAL_DONE_quick.txt" if q else "EVAL_DONE.txt")
    hub_repo = None if (args.no_push or (q and not args.push_to_hub)) else (args.push_to_hub or DEFAULT_HUB_REPO)
    changed = False            # did anything change in this run? then evaluate again
    t0 = time.time()

    # 1 --------------------------------------------------------------------------
    run("1/7 Check GPU", "scripts/check_gpu.py")
    if hub_repo:
        check_hub(hub_repo)

    # 2 --------------------------------------------------------------------------
    data_version = f"scale={scale} cleaning=khmer-only-v1 sources=lm-corpus+real-typos-v2"
    built = DATA_VERSION_FILE.read_text().strip() if DATA_VERSION_FILE.exists() else None
    new_data = args.rebuild_data or built != data_version
    if new_data:
        run("2/7 Prepare data", "scripts/prepare_data.py", "--scale", scale)
        DATA_VERSION_FILE.write_text(data_version)
        changed = True
    else:
        skip("2/7 Prepare data", f"already prepared ({data_version})")

    # 3 --------------------------------------------------------------------------
    if new_data or not config.LEXICON_TSV.exists():
        run("3/7 Build lexicon", "scripts/build_lexicon.py")
    else:
        skip("3/7 Build lexicon", "up to date")

    # 4 --------------------------------------------------------------------------
    if args.retrain or not config.G2P_CKPT.exists():
        run("4/7 Train G2P", "scripts/train_g2p.py", "--epochs", 30)
        changed = True
    else:
        skip("4/7 Train G2P", "g2p.pt exists")
    if changed or not config.PRON_TSV.exists():
        run("4/7 Pronunciations of the lexicon words", "scripts/build_pron.py")

    # 5 --------------------------------------------------------------------------
    if args.retrain or not tagger_pt.exists():
        run("5/7 Train character tagger", "scripts/train_tagger.py", "--out", tagger_pt,
            *(["--epochs", 1, "--limit", 60000] if q else []))
        changed = True
    else:
        skip("5/7 Train character tagger", f"{tagger_pt.name} exists")

    # 6 --------------------------------------------------------------------------
    if args.retrain or not (topk_dir / "config.json").exists():
        train_args = ["--model", args.model, "--max-steps", steps, "--out", topk_dir]
        if not args.retrain and any(topk_dir.glob("checkpoint-*")):
            train_args.append("--resume")
            print("\n(checkpoints found: ByT5 training continues where it stopped)")
        if q:
            train_args.append("--scale-test")
        if hub_repo:
            train_args += ["--push-to-hub", hub_repo]
        run("6/7 Train ByT5 model", "scripts/train_topk.py", *train_args)
        changed = True
    else:
        skip("6/7 Train ByT5 model", f"{topk_dir.name} is finished")

    # 7 --------------------------------------------------------------------------
    if changed or not eval_done.exists():
        eval_args = ["--model", topk_dir, "--tagger", tagger_pt, "--tune"]
        if q:
            eval_args += ["--n-tune", 200, "--n-words", 300, "--n-sent", 100, "--n-real", 300]
        run("7/7 Tune ranking + evaluate", "scripts/eval_topk.py", *eval_args)
        eval_done.write_text(time.ctime())
        if hub_repo:
            print(f"\n>>> Uploading tagger, lexicon, G2P and ranking weights to {hub_repo}/speller/")
            upload_extras(hub_repo, [tagger_pt, config.LEXICON_TSV, config.PRON_TSV, config.G2P_CKPT,
                                     config.RANK_WEIGHTS])
    else:
        skip("7/7 Tune ranking + evaluate", f"nothing changed; results are in {config.LOG}")

    print(f"\nALL DONE in {(time.time() - t0) / 3600:.1f} h.   Results: {config.LOG}")
    os.environ["KHSPELL_TOPK_MODEL"] = str(topk_dir)
    os.environ["KHSPELL_TAGGER"] = str(tagger_pt)
    if args.serve:
        print("API: http://localhost:8000/docs   (Ctrl+C to stop)")
        subprocess.run([sys.executable, "-m", "uvicorn", "app:app", "--port", "8000"], cwd=ROOT)
    else:
        print(f"Start the API:  python main.py --serve{' --quick' if q else ''}")


if __name__ == "__main__":
    main()
