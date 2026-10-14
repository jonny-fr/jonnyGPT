#!/usr/bin/env python3
"""Small, randomly initialized Llama: prepare -> train -> sample."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
SOURCES = {
    "web": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus"),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def write_json(path, obj):
    def clean(value):
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if not callable(item)
            }

        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]

        return value

    Path(path).write_text(
        json.dumps(clean(obj), indent=2),
        encoding="utf-8",
    )


def local_rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prepare(args):
    root = Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise SystemExit("Data directory already prepared; choose a new --data directory.")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token.")
    vocab = len(tok)
    dtype = np.dtype("uint16" if vocab <= 65536 else "uint32")
    weights = {"web": .8, "code": .1, "math": .1} if args.code_jsonl else {"web": .9, "math": .1}
    seen = set()
    info = {}
    start = time.monotonic()
    for index, (name, weight) in enumerate(weights.items()):
        train_target = max(args.seq_len + 1, int(args.train_tokens * weight))
        val_target = args.val_tokens
        if name == "code":
            rows = local_rows(args.code_jsonl)
            source_description = {"local_jsonl": str(Path(args.code_jsonl).resolve())}
        else:
            repo, config = SOURCES[name]
            rows = load_dataset(repo, config, split="train", streaming=True)
            rows = rows.shuffle(seed=args.seed + index, buffer_size=args.shuffle_buffer)
            source_description = {"repo": repo, "config": config}
        counts = {"train": 0, "val": 0}
        docs = {"train": 0, "val": 0}
        paths = {s: root / f"{name}_{s}.bin" for s in counts}
        with contextlib.ExitStack() as stack:
            files = {s: stack.enter_context(open(p, "wb")) for s, p in paths.items()}
            for row in rows:
                text = row.get("text") or row.get("content") or row.get("code")
                if not isinstance(text, str) or len(text.strip()) < 80:
                    continue
                normalized = text.strip()
                digest = hashlib.sha256(normalized.encode("utf-8")).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                # Exact duplicates stay out across all sources and both splits.
                split = "val" if int.from_bytes(digest[:8], "little") % 100 < 2 else "train"
                target = val_target if split == "val" else train_target
                if counts[split] >= target:
                    continue
                ids = tok(normalized[:args.max_chars], add_special_tokens=False,
                          truncation=True, max_length=args.max_doc_tokens)["input_ids"]
                ids.append(tok.eos_token_id)
                ids = ids[:target - counts[split]]
                np.asarray(ids, dtype=dtype).tofile(files[split])
                counts[split] += len(ids)
                docs[split] += 1
                if docs[split] % 1000 == 0:
                    print(name, counts, flush=True)
                if counts["train"] >= train_target and counts["val"] >= val_target:
                    break
        if min(counts.values()) < args.seq_len + 1:
            raise RuntimeError(f"Insufficient {name} data: {counts}; supply more documents.")
        info[name] = {"weight": weight, "tokens": counts, "documents": docs,
                      "source": source_description}
        print(f"Prepared {name}: {counts}", flush=True)
    tok.save_pretrained(root / "tokenizer")
    write_json(root / "manifest.json", {"dtype": dtype.name, "vocab_size": vocab,
               "eos_token_id": tok.eos_token_id, "sources": info, "args": vars(args),
               "preparation_seconds": time.monotonic() - start})


class TokenStore:
    def __init__(self, root, seq_len, seed):
        self.root = Path(root)
        self.meta = json.loads((self.root / "manifest.json").read_text())
        self.seq_len = seq_len
        self.names = list(self.meta["sources"])
        self.weights = np.array([self.meta["sources"][n]["weight"] for n in self.names])
        self.weights /= self.weights.sum()
        self.rng = np.random.default_rng(seed)
        self.arrays = {(n, s): np.memmap(self.root / f"{n}_{s}.bin", mode="r",
                      dtype=self.meta["dtype"]) for n in self.names for s in ("train", "val")}
        for key, arr in self.arrays.items():
            if len(arr) < seq_len + 1:
                raise ValueError(f"Too little data in {key} for sequence length {seq_len}.")

    def batch(self, name, split, batch_size, rng=None):
        rng = self.rng if rng is None else rng
        arr = self.arrays[name, split]
        starts = rng.integers(0, len(arr) - self.seq_len, size=batch_size)
        batch = np.stack([arr[s:s+self.seq_len+1] for s in starts]).astype(np.int64)
        x = torch.from_numpy(batch[:, :-1].copy()).pin_memory().to("cuda", non_blocking=True)
        y = torch.from_numpy(batch[:, 1:].copy()).pin_memory().to("cuda", non_blocking=True)
        return x, y


def amp_context(dtype):
    return torch.autocast(device_type="cuda", dtype=dtype)


def loss_for(model, x, y):
    # Inputs and targets are already shifted; do NOT also pass labels to the model.
    logits = model(input_ids=x, use_cache=False).logits
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def evaluate(model, store, batch_size, batches, dtype):
    model.eval()
    values = {}
    rng = np.random.default_rng(987654)
    for name in store.names:
        total = 0.
        for _ in range(batches):
            x, y = store.batch(name, "val", batch_size, rng)
            with amp_context(dtype):
                loss = loss_for(model, x, y)
            total += loss.item()
        values[name] = total / batches
    values["weighted"] = sum(values[n] * w for n, w in zip(store.names, store.weights))
    model.train()
    return values


def lr_factor(step, total, warmup=.05, decay=.20):
    warmup_steps = max(1, round(total * warmup))
    decay_start = max(warmup_steps, round(total * (1 - decay)))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step < decay_start:
        return 1.
    return max(0., (total - 1 - step) / max(1, total - 1 - decay_start))


def train(args):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Install a CUDA-enabled PyTorch build.")
    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists() or (root / "run.json").exists():
        raise SystemExit("Output already contains a run; choose a new --out directory.")
    store = TokenStore(args.data, args.seq_len, args.seed)
    tok = AutoTokenizer.from_pretrained(Path(args.data) / "tokenizer")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config = LlamaConfig(vocab_size=len(tok), hidden_size=384, intermediate_size=1024,
             num_hidden_layers=8, num_attention_heads=6, num_key_value_heads=6,
             max_position_embeddings=args.seq_len, rope_theta=10000.,
             hidden_act="silu", tie_word_embeddings=True, attention_dropout=0.,
             eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id,
             pad_token_id=tok.eos_token_id, use_cache=False)
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).cuda()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                 betas=(.9, .95), weight_decay=.1, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    nparams = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.batch_size * args.accumulation * args.seq_len
    print(f"GPU: {torch.cuda.get_device_name()} | parameters: {nparams:,} | dtype: {dtype}")
    print(f"Tokens per optimizer step: {tokens_per_step:,}", flush=True)
    model.train()

    def update(lr):
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device="cuda")
        for _ in range(args.accumulation):
            name = store.rng.choice(store.names, p=store.weights)
            x, y = store.batch(name, "train", args.batch_size)
            with amp_context(dtype):
                loss = loss_for(model, x, y)
            total += loss.detach() / args.accumulation
            scaler.scale(loss / args.accumulation).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        return total.item()

    wall_start = time.monotonic()
    deadline = wall_start + args.hours * 3600
    # Dry updates use LR=0, then reset optimizer/scaler/RNG: weights are unchanged.
    for i in range(3):
        torch.cuda.synchronize()
        t = time.monotonic()
        update(0.)
        torch.cuda.synchronize()
        if i == 1:
            bench_start = t
    elapsed = time.monotonic() - bench_start
    seconds_per_step = elapsed / 2
    optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    seed_all(args.seed)
    store.rng = np.random.default_rng(args.seed)
    remaining = max(0., deadline - time.monotonic())
    steps = int(remaining * .80 / seconds_per_step)
    if args.max_steps is not None:
        steps = min(steps, args.max_steps)
    if steps < 10:
        raise SystemExit("Time budget too short; increase --hours.")
    run = {"args": vars(args), "parameters": nparams, "dtype": str(dtype),
           "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
           "seconds_per_step_benchmark": seconds_per_step, "planned_steps": steps,
           "planned_tokens": steps * tokens_per_step, "data": store.meta}
    write_json(root / "run.json", run)
    print(f"Benchmark: {tokens_per_step / seconds_per_step:,.0f} training tokens/s")
    print(f"Plan: {steps:,} steps, {steps*tokens_per_step:,} tokens; 20% time reserve.", flush=True)
    completed = 0
    stopped = "completed"
    start_training = time.monotonic()
    try:
        with open(root / "metrics.csv", "w", newline="") as f:
            fields = ["step", "tokens", "seconds", "lr", "train_loss", "val_weighted"] + [f"val_{n}" for n in store.names]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            eval_seconds_start = time.monotonic()
            baseline = evaluate(model, store, args.batch_size, args.eval_batches, dtype)
            eval_seconds = time.monotonic() - eval_seconds_start
            writer.writerow({"step": 0, "tokens": 0, "seconds": time.monotonic()-wall_start,
                              **{f"val_{k}": v for k, v in baseline.items()}})
            f.flush()
            print("Initial validation:", baseline, flush=True)
            last_eval = time.monotonic()
            for step in range(steps):
                if time.monotonic() + seconds_per_step + 15 > deadline:
                    stopped = "time_limit"
                    break
                lr = args.lr * lr_factor(step, steps)
                loss = update(lr)
                completed = step + 1
                if not math.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss; reduce learning rate.")
                row = {"step": completed, "tokens": completed*tokens_per_step,
                       "seconds": time.monotonic()-wall_start, "lr": lr, "train_loss": loss}
                if time.monotonic()-last_eval >= args.eval_every_minutes*60 or completed == steps:
                    if time.monotonic()+eval_seconds+15 < deadline:
                        val = evaluate(model, store, args.batch_size, args.eval_batches, dtype)
                        row.update({f"val_{k}": v for k,v in val.items()})
                        print("Validation:", val, flush=True)
                        last_eval = time.monotonic()
                writer.writerow(row)
                f.flush()
                if completed % args.log_every == 0:
                    throughput = completed*tokens_per_step / (time.monotonic()-start_training)
                    print(f"{completed}/{steps} | loss {loss:.4f} | lr {lr:.2e} | {throughput:,.0f} tok/s", flush=True)
    except KeyboardInterrupt:
        stopped = "keyboard_interrupt"
        print("Interrupted; exporting current model...", flush=True)
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.config.use_cache = True
        model.save_pretrained(root, safe_serialization=True)
        tok.save_pretrained(root)
        write_json(root / "result.json", {"completed_steps": completed,
                   "training_tokens": completed*tokens_per_step, "stop_reason": stopped,
                   "elapsed_seconds": time.monotonic()-wall_start,
                   "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated()/1024**3})
        print(f"Model exported to {root}", flush=True)


def sample(args):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this example.")
    tok = AutoTokenizer.from_pretrained(args.out)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = LlamaForCausalLM.from_pretrained(args.out, torch_dtype=dtype,
            attn_implementation="sdpa").cuda().eval()
    x = tok(args.prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    limit = model.config.max_position_embeddings
    remaining = limit - x.input_ids.shape[1]
    if remaining <= 0:
        raise SystemExit(f"Prompt must contain fewer than {limit} tokens.")
    with torch.inference_mode():
        y = model.generate(**x, max_new_tokens=min(args.new_tokens, remaining),
                do_sample=True, temperature=.8, top_p=.95, repetition_penalty=1.1,
                pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    print(tok.decode(y[0], skip_special_tokens=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--data", default="mini_data")
    prep.add_argument("--train-tokens", type=int, default=60_000_000)
    prep.add_argument("--val-tokens", type=int, default=131_072, help="Per source")
    prep.add_argument("--code-jsonl", help="Local JSONL with text/content/code field")
    prep.add_argument("--seq-len", type=int, default=512)
    prep.add_argument("--shuffle-buffer", type=int, default=1000)
    prep.add_argument("--max-chars", type=int, default=40_000)
    prep.add_argument("--max-doc-tokens", type=int, default=8192)
    prep.add_argument("--seed", type=int, default=42)
    prep.set_defaults(func=prepare)
    tr = subs.add_parser("train")
    tr.add_argument("--data", default="mini_data")
    tr.add_argument("--out", default="mini_model")
    tr.add_argument("--hours", type=float, default=3.)
    tr.add_argument("--seq-len", type=int, default=512)
    tr.add_argument("--batch-size", type=int, default=4)
    tr.add_argument("--accumulation", type=int, default=8)
    tr.add_argument("--lr", type=float, default=5e-4)
    tr.add_argument("--eval-batches", type=int, default=8)
    tr.add_argument("--eval-every-minutes", type=float, default=10.)
    tr.add_argument("--log-every", type=int, default=20)
    tr.add_argument("--max-steps", type=int)
    tr.add_argument("--gradient-checkpointing", action="store_true")
    tr.add_argument("--seed", type=int, default=42)
    tr.set_defaults(func=train)
    sm = subs.add_parser("sample")
    sm.add_argument("--out", default="mini_model")
    sm.add_argument("--prompt", default="The solar system consists of")
    sm.add_argument("--new-tokens", type=int, default=150)
    sm.set_defaults(func=sample)
    args = p.parse_args()
    for key in ("train_tokens", "val_tokens", "seq_len", "shuffle_buffer", "max_chars",
                "max_doc_tokens", "batch_size", "accumulation", "eval_batches", "log_every",
                "hours", "eval_every_minutes", "max_steps", "new_tokens"):
        if hasattr(args, key) and getattr(args, key) is not None and getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
