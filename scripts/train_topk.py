"""Train the word-in-context top-k speller (one seq2seq model).

Needs outputs/data/train.parquet from scripts/prepare_data.py.

    python scripts/train_topk.py                                   # RTX 3090: ~2-3 h
    python scripts/train_topk.py --max-steps 3000 --scale-test     # quick check (~15 min)

Each training example is one word of a sentence, misspelled 80% of the time, with
its surrounding text as context. New errors are drawn every time (khspell.noise).
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

from khspell import config
from khspell.topk import make_example



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/byt5-small")
    ap.add_argument("--out", default=str(config.TOPK_DIR))
    ap.add_argument("--max-steps", type=int, default=30_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--per-device-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval-steps", type=int, default=2000)
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--scale-test", action="store_true", help="use 50k sentences only")
    ap.add_argument("--push-to-hub", default=None, help="e.g. phonsobon/khmer-spell-checker (public)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    df = pd.read_parquet(config.TRAIN_DATA, columns=["tokens"])
    if args.scale_test:
        df = df.head(50_000)
    dev_df, train_df = df.iloc[:args.n_dev], df.iloc[args.n_dev:]

    # fixed dev examples (same misspellings every evaluation)
    drng = random.Random(0)
    dev_pairs = [p for p in (make_example(list(t), drng) for t in dev_df["tokens"]) if p]
    print(f"train {len(train_df):,} sentences | dev {len(dev_pairs):,} words")

    def encode(srcs, tgts):
        enc = tok(srcs, max_length=384, truncation=True)
        enc["labels"] = tok(text_target=tgts, max_length=96, truncation=True)["input_ids"]
        return enc

    rng = random.Random(args.seed)

    def transform(batch):
        pairs = [make_example(list(t), rng) for t in batch["tokens"]]
        pairs = [p if p else ("[[។]]", "។") for p in pairs]
        return encode([s for s, _ in pairs], [g for _, g in pairs])

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    train_ds.set_transform(transform)
    dev_ds = Dataset.from_dict(encode([s for s, _ in dev_pairs], [g for _, g in dev_pairs]))
    gold = [g for _, g in dev_pairs]

    def compute_metrics(pred):
        preds = np.where(pred.predictions < 0, tok.pad_token_id, pred.predictions)
        hyp = tok.batch_decode(preds, skip_special_tokens=True)
        return {"acc@1": float(np.mean([h == g for h, g in zip(hyp, gold)]))}

    bf16 = config.bf16_ok()                      # T4: fp32 (never fp16 for T5-family models)
    per_dev = min(args.per_device_batch, args.batch)
    n_proc = int(os.environ.get("WORLD_SIZE", 1))  # GPUs under torchrun: --batch stays the total batch
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        max_steps=args.max_steps,
        per_device_train_batch_size=per_dev,
        per_device_eval_batch_size=per_dev,
        gradient_accumulation_steps=max(1, args.batch // (per_dev * n_proc)),
        ddp_find_unused_parameters=False,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=min(1000, args.max_steps // 10),
        weight_decay=0.01,
        bf16=bf16,                               # never fp16 for T5-family models
        eval_strategy="steps",
        eval_steps=min(args.eval_steps, args.max_steps),
        save_steps=min(args.eval_steps, args.max_steps),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="acc@1",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=96,
        logging_steps=100,
        dataloader_num_workers=0,                # the on-the-fly transform uses a shared RNG
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
        hub_private_repo=False,
        hub_strategy="checkpoint",               # every save also uploads last-checkpoint/ (for resume)
    )
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model), processing_class=tok,
                             compute_metrics=compute_metrics)
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(args.out)
    if trainer.is_world_process_zero():
        tok.save_pretrained(args.out)
    print(trainer.evaluate())
    if args.push_to_hub:
        for attempt in range(1, 6):              # the model is saved locally already: only retry the upload
            try:
                trainer.push_to_hub()
                break
            except Exception as e:
                print(f"upload failed ({type(e).__name__}: {e}), retry {attempt}/5 in {30 * attempt}s", flush=True)
                time.sleep(30 * attempt)


if __name__ == "__main__":
    main()
