#!/usr/bin/env python3
"""Train a randomly initialized 66M Llama: prepare -> verify -> train -> sample."""
import argparse
import contextlib
import csv
import hashlib
import json
import logging
import math
import os
import random
import signal
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from datasets import load_dataset
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn, Progress, TaskProgressColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
SOURCES = {
    "web": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "math": ("HuggingFaceTB/finemath", "finemath-4plus"),
}
CONSOLE = Console(stderr=True)
LOG = logging.getLogger("mini_smollm")
TRAIN_DEFAULTS = {
    "seq_len": 512, "batch_size": 8, "accumulation": 4,
    "lr": 5e-4, "eval_batch_size": 8, "eval_batches": 64,
    "eval_every_minutes": 15.0, "log_every": 100,
    "checkpoint_every_minutes": 10.0, "gradient_checkpointing": False,
    "seed": 42, "warmup": .05, "decay": .20, "max_steps": None,
}


def configure_logging(path, level):
    handler = RichHandler(
        console=CONSOLE, show_time=True, show_level=True, show_path=False,
        log_time_format="[%Y-%m-%d %H:%M:%S]", omit_repeated_times=False,
        rich_tracebacks=True, tracebacks_show_locals=False, markup=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.basicConfig(
        level=getattr(logging, level), handlers=[handler, file_handler], force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    LOG.info("Log file: %s", path)


def duration(seconds):
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(clean_json(value), handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(store):
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "store": store.rng.bit_generator.state,
    }


def restore_rng(state, store):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    store.rng.bit_generator.state = state["store"]


def progress_bar():
    return Progress(
        TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(),
        TextColumn("{task.completed:,.0f}/{task.total:,.0f} tokens"),
        TimeElapsedColumn(), TimeRemainingColumn(), console=CONSOLE,
        disable=not CONSOLE.is_terminal,
    )


def local_rows(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prepare(args):
    root = Path(args.data).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError("Directory already prepared; choose a new --data directory.")
    marker = root / ".prepare_incomplete"
    if marker.exists() or list(root.glob("*.bin*")) or (root / "tokenizer").exists():
        raise ValueError("Incomplete data already present. Inspect it or use a new directory.")
    marker.write_text(datetime.now().isoformat(), encoding="utf-8")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token.")
    dtype = np.dtype("uint16" if len(tok) <= 65536 else "uint32")
    weights = {"web": .8, "code": .1, "math": .1} if args.code_jsonl else {"web": .9, "math": .1}
    if any(int(args.train_tokens * w) < args.seq_len + 1 for w in weights.values()):
        raise ValueError("Token target too small for at least one source.")
    seen, info = set(), {}
    started = time.monotonic()
    LOG.info("Preparing %s training tokens; %s validation tokens PER source",
             f"{args.train_tokens:,}", f"{args.val_tokens:,}")
    with progress_bar() as progress:
        for index, (name, weight) in enumerate(weights.items()):
            targets = {"train": int(args.train_tokens * weight), "val": args.val_tokens}
            if name == "code":
                rows = local_rows(args.code_jsonl)
                description = {"local_jsonl": str(Path(args.code_jsonl).resolve())}
            else:
                repo, config = SOURCES[name]
                LOG.info("Loading %s: %s / %s", name, repo, config)
                rows = load_dataset(repo, config, split="train", streaming=True)
                rows = rows.shuffle(seed=args.seed + index, buffer_size=args.shuffle_buffer)
                description = {"repo": repo, "config": config}
            counts = {"train": 0, "val": 0}
            docs = {"train": 0, "val": 0}
            hashes = {s: hashlib.sha256() for s in targets}
            paths = {s: root / f"{name}_{s}.bin" for s in targets}
            tasks = {s: progress.add_task(f"{name}/{s}", total=targets[s]) for s in targets}
            last_log = time.monotonic()
            with contextlib.ExitStack() as stack:
                files = {s: stack.enter_context(open(str(p) + ".tmp", "wb")) for s, p in paths.items()}
                for row in rows:
                    text = row.get("text") or row.get("content") or row.get("code")
                    if not isinstance(text, str) or len(text.strip()) < 80:
                        continue
                    normalized = text.strip()
                    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    split = "val" if int.from_bytes(digest[:8], "little") % 100 < 2 else "train"
                    if counts[split] >= targets[split]:
                        continue
                    ids = tok(
                        normalized[:args.max_chars], add_special_tokens=False,
                        truncation=True, max_length=args.max_doc_tokens,
                    )["input_ids"]
                    ids.append(tok.eos_token_id)
                    ids = ids[:targets[split] - counts[split]]
                    data = np.asarray(ids, dtype=dtype).tobytes()
                    files[split].write(data)
                    hashes[split].update(data)
                    counts[split] += len(ids)
                    docs[split] += 1
                    progress.update(tasks[split], advance=len(ids))
                    if time.monotonic() - last_log >= 30:
                        elapsed = time.monotonic() - started
                        LOG.info("Prepare %s | train=%s | val=%s | elapsed=%s",
                                 name, f"{counts['train']:,}", f"{counts['val']:,}", duration(elapsed))
                        last_log = time.monotonic()
                    if all(counts[s] >= targets[s] for s in targets):
                        break
                for handle in files.values():
                    handle.flush()
                    os.fsync(handle.fileno())
            if counts != targets:
                raise RuntimeError(f"Source {name} exhausted: got {counts}, wanted {targets}.")
            for split, path in paths.items():
                temporary = Path(str(path) + ".tmp")
                if temporary.stat().st_size != counts[split] * dtype.itemsize:
                    raise RuntimeError(f"Incorrect file size: {temporary}")
                os.replace(temporary, path)
            info[name] = {
                "weight": weight, "tokens": counts, "documents": docs,
                "source": description,
                "sha256": {s: h.hexdigest() for s, h in hashes.items()},
            }
            LOG.info("Prepared %s | train=%s | val=%s", name, f"{counts['train']:,}", f"{counts['val']:,}")
    tok.save_pretrained(root / "tokenizer")
    write_json(root / "manifest.json", {
        "dtype": dtype.name, "vocab_size": len(tok), "eos_token_id": tok.eos_token_id,
        "sources": info, "args": vars(args),
        "preparation_seconds": time.monotonic() - started,
    })
    marker.unlink()
    LOG.info("Preparation complete in %s: %s", duration(time.monotonic() - started), root)


class TokenStore:
    def __init__(self, root, seq_len, seed):
        self.root = Path(root).resolve()
        self.meta = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.seq_len = seq_len
        self.names = list(self.meta["sources"])
        if self.meta["dtype"] not in ("uint16", "uint32"):
            raise ValueError("Unsupported token storage dtype.")
        if not self.names:
            raise ValueError("No data sources.")
        self.weights = np.array([self.meta["sources"][n]["weight"] for n in self.names], dtype=float)
        if not np.isfinite(self.weights).all() or (self.weights <= 0).any():
            raise ValueError("Source weights must be finite and positive.")
        self.weights /= self.weights.sum()
        self.rng = np.random.default_rng(seed)
        self.arrays = {}
        for name in self.names:
            for split in ("train", "val"):
                path = self.root / f"{name}_{split}.bin"
                expected = self.meta["sources"][name]["tokens"][split]
                if expected < seq_len + 1:
                    raise ValueError(f"Too little data in {path}.")
                if path.stat().st_size != expected * np.dtype(self.meta["dtype"]).itemsize:
                    raise ValueError(f"File size does not match manifest: {path}")
                self.arrays[name, split] = np.memmap(path, mode="r", dtype=self.meta["dtype"])

    def verify(self):
        fingerprint = hashlib.sha256()
        fingerprint.update(json.dumps(self.meta, sort_keys=True).encode())
        for (name, split), arr in self.arrays.items():
            digest = hashlib.sha256()
            for start in range(0, len(arr), 4_194_304):
                chunk = arr[start:start + 4_194_304]
                if int(chunk.max()) >= self.meta["vocab_size"]:
                    raise ValueError(f"Out-of-range token ID in {name}/{split}.")
                digest.update(chunk.tobytes())
            actual = digest.hexdigest()
            expected = self.meta["sources"][name].get("sha256", {}).get(split)
            if expected and actual != expected:
                raise ValueError(f"Checksum mismatch: {name}/{split}")
            fingerprint.update(f"{name}/{split}:{actual}".encode())
            LOG.info("Verified %s/%s: %s tokens", name, split, f"{len(arr):,}")
        tokenizer_files = sorted(p for p in (self.root / "tokenizer").rglob("*") if p.is_file())
        if not tokenizer_files:
            raise ValueError("Missing tokenizer files.")
        for path in tokenizer_files:
            fingerprint.update(str(path.relative_to(self.root / "tokenizer")).encode())
            fingerprint.update(path.read_bytes())
        return fingerprint.hexdigest()

    def batch(self, name, split, batch_size, rng=None):
        rng = self.rng if rng is None else rng
        arr = self.arrays[name, split]
        starts = rng.integers(0, len(arr) - self.seq_len, size=batch_size)
        batch = np.stack([arr[s:s + self.seq_len + 1] for s in starts]).astype(np.int64)
        x = torch.from_numpy(batch[:, :-1].copy()).pin_memory().to("cuda", non_blocking=True)
        y = torch.from_numpy(batch[:, 1:].copy()).pin_memory().to("cuda", non_blocking=True)
        return x, y


def verify(args):
    store = TokenStore(args.data, args.seq_len, args.seed)
    fingerprint = store.verify()
    tok = AutoTokenizer.from_pretrained(store.root / "tokenizer")
    if len(tok) != store.meta["vocab_size"] or tok.eos_token_id != store.meta["eos_token_id"]:
        raise ValueError("Tokenizer does not match manifest.")
    LOG.info("Data verification successful | fingerprint=%s", fingerprint)


def loss_for(model, x, y):
    logits = model(input_ids=x, use_cache=False).logits
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def evaluate(model, store, batch_size, batches, dtype):
    previous_mode = model.training
    model.eval()
    values = {}
    try:
        for index, name in enumerate(store.names):
            rng = np.random.default_rng(987654 + index)
            total = 0.0
            for _ in range(batches):
                x, y = store.batch(name, "val", batch_size, rng)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    loss = loss_for(model, x, y)
                total += loss.item()
            values[name] = total / batches
        values["weighted"] = float(sum(values[n] * w for n, w in zip(store.names, store.weights)))
        if not all(math.isfinite(v) for v in values.values()):
            raise FloatingPointError("Non-finite validation loss.")
        return values
    finally:
        model.train(previous_mode)


def lr_factor(step, total, warmup=.05, decay=.20):
    warmup_steps = max(1, round(total * warmup))
    decay_start = max(warmup_steps, round(total * (1 - decay)))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step < decay_start:
        return 1.0
    return max(0.0, (total - 1 - step) / max(1, total - 1 - decay_start))


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format; use checkpoints from this script.")
    return checkpoint


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        settings = dict(checkpoint["settings"])
        root = Path(checkpoint["output_directory"])
        if args.out and Path(args.out).resolve() != root:
            raise ValueError("Resume must use the original --out directory.")
        if args.data and Path(args.data).resolve() != Path(settings["data"]):
            raise ValueError("Resume must use the original --data directory.")
        operational = {"log_every", "checkpoint_every_minutes", "eval_every_minutes"}
        for key in TRAIN_DEFAULTS:
            requested = getattr(args, key)
            if requested is not None:
                if key not in operational and requested != settings[key]:
                    raise ValueError(f"Cannot change --{key.replace('_', '-')} during resume.")
                settings[key] = requested
    else:
        settings = {k: TRAIN_DEFAULTS[k] if getattr(args, k) is None else getattr(args, k)
                    for k in TRAIN_DEFAULTS}
        settings["data"] = str(Path(args.data or "mini_data").resolve())
        root = Path(args.out or "mini_model").resolve()
        root.mkdir(parents=True, exist_ok=True)
        if any((root / name).exists() for name in ("run.json", "config.json", "checkpoint_last.pt")):
            raise ValueError("Output already contains a run. Use --resume or a new --out.")
    if not 0 < settings["warmup"] < 1 or not 0 < settings["decay"] < 1:
        raise ValueError("Warmup and decay must each lie between 0 and 1.")
    if settings["warmup"] + settings["decay"] >= 1:
        raise ValueError("Warmup + decay must be less than 1.")
    root.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    seed_all(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    store = TokenStore(settings["data"], settings["seq_len"], settings["seed"])
    LOG.info("Checking dataset sizes, token IDs and SHA-256 checksums...")
    data_fingerprint = store.verify()
    tok = AutoTokenizer.from_pretrained(Path(settings["data"]) / "tokenizer")
    if len(tok) != store.meta["vocab_size"] or tok.eos_token_id != store.meta["eos_token_id"]:
        raise ValueError("Tokenizer does not match manifest.")
    if checkpoint and data_fingerprint != checkpoint["data_fingerprint"]:
        raise ValueError("Dataset or tokenizer changed since checkpoint.")
    if checkpoint:
        dtype_name = checkpoint["dtype"]
        if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Checkpoint requires BF16 support.")
        dtype = getattr(torch, dtype_name)
        config = LlamaConfig.from_dict(checkpoint["config"])
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        dtype_name = str(dtype).split(".")[-1]
        config = LlamaConfig(
            vocab_size=len(tok), hidden_size=512, intermediate_size=1536,
            num_hidden_layers=12, num_attention_heads=8, num_key_value_heads=8,
            max_position_embeddings=settings["seq_len"], rope_theta=10000.0,
            hidden_act="silu", tie_word_embeddings=True, attention_dropout=0.0,
            eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id,
            pad_token_id=tok.eos_token_id, use_cache=False,
        )
    config.use_cache = False
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).cuda()
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    if settings["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], betas=(.9, .95), weight_decay=.1, fused=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
    nparams = sum(p.numel() for p in model.parameters())
    tokens_per_step = settings["batch_size"] * settings["accumulation"] * settings["seq_len"]
    LOG.info("GPU: %s | parameters=%s | dtype=%s", torch.cuda.get_device_name(), f"{nparams:,}", dtype_name)
    LOG.info("Microbatch=%d | accumulation=%d | context=%d | tokens/update=%s",
             settings["batch_size"], settings["accumulation"], settings["seq_len"], f"{tokens_per_step:,}")
    LOG.info("Evaluation: %d x %d x %d = %s token positions per source",
             settings["eval_batches"], settings["eval_batch_size"], settings["seq_len"],
             f"{settings['eval_batches'] * settings['eval_batch_size'] * settings['seq_len']:,}")
    model.train()

    def update(lr):
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device="cuda")
        for _ in range(settings["accumulation"]):
            name = store.rng.choice(store.names, p=store.weights)
            x, y = store.batch(name, "train", settings["batch_size"])
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = loss_for(model, x, y)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite loss; last saved checkpoint was not replaced.")
            total += loss.detach() / settings["accumulation"]
            scaler.scale(loss / settings["accumulation"]).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=(dtype == torch.bfloat16),
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.is_enabled() and scaler.get_scale() < old_scale
        return total.item(), float(norm), skipped

    session_start = time.monotonic()
    deadline = session_start + args.hours * 3600
    stop_request = {"reason": None}

    def request_stop(signum, frame):
        stop_request["reason"] = signal.Signals(signum).name

    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, request_stop)
    try:
        if checkpoint:
            completed = checkpoint["completed_steps"]
            planned_steps = checkpoint["planned_steps"]
            seconds_per_step = checkpoint["seconds_per_step"]
            best_val = checkpoint["best_val"]
            prior_elapsed = checkpoint["total_elapsed_seconds"]
            run_id = checkpoint["run_id"]
            restore_rng(checkpoint["rng"], store)
            LOG.info("Resuming step %s/%s; original LR schedule is preserved",
                     f"{completed:,}", f"{planned_steps:,}")
            del checkpoint
            if completed >= planned_steps:
                raise ValueError("Training plan is complete. Resume does not extend the LR schedule.")
        else:
            LOG.info("Benchmark: 1 warmup update + 10 timed updates, learning rate zero")
            update(0.0)
            torch.cuda.synchronize()
            benchmark_start = time.monotonic()
            for _ in range(10):
                update(0.0)
            torch.cuda.synchronize()
            seconds_per_step = (time.monotonic() - benchmark_start) / 10
            optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)
            scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
            seed_all(settings["seed"])
            store.rng = np.random.default_rng(settings["seed"])
            remaining = max(0, deadline - time.monotonic())
            planned_steps = int(remaining * .80 / seconds_per_step)
            if settings["max_steps"] is not None:
                planned_steps = min(planned_steps, settings["max_steps"])
            if planned_steps < 10:
                raise ValueError("Time budget too short.")
            completed, best_val, prior_elapsed = 0, float("inf"), 0.0
            run_id = uuid4().hex
            tok.save_pretrained(root / "tokenizer")
            write_json(root / "run.json", {
                "run_id": run_id, "settings": settings, "parameters": nparams,
                "config": model.config.to_dict(), "dtype": dtype_name,
                "torch_version": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
                "seconds_per_step_benchmark": seconds_per_step,
                "planned_steps": planned_steps, "planned_tokens": planned_steps * tokens_per_step,
                "data_fingerprint": data_fingerprint, "data": store.meta,
            })
        LOG.info("Benchmark: %s tokens/s | plan: %s steps / %s tokens",
                 f"{tokens_per_step / seconds_per_step:,.0f}", f"{planned_steps:,}",
                 f"{planned_steps * tokens_per_step:,}")
        initial_completed = completed
        train_seconds, session_train_tokens = 0.0, 0
        smooth_seconds = seconds_per_step
        interval_losses = []
        last_eval = time.monotonic()
        last_save = time.monotonic()
        last_val_step = None
        checkpoint_seconds = 0.0
        decay_start = max(max(1, round(planned_steps * settings["warmup"])),
                          round(planned_steps * (1 - settings["decay"])))
        metrics_path = root / (f"metrics_resume_{session_id}.csv" if args.resume else "metrics.csv")
        write_json(root / f"session_{session_id}.json", {
            "resume_from": str(Path(args.resume).resolve()) if args.resume else None,
            "initial_step": completed, "hours": args.hours, "settings": settings,
            "metrics": metrics_path.name,
        })

        def snapshot():
            return {
                "format_version": 1, "run_id": run_id,
                "output_directory": str(root), "settings": settings,
                "config": model.config.to_dict(), "dtype": dtype_name,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng_state(store),
                "completed_steps": completed, "planned_steps": planned_steps,
                "training_tokens": completed * tokens_per_step,
                "seconds_per_step": smooth_seconds, "best_val": best_val,
                "data_fingerprint": data_fingerprint,
                "total_elapsed_seconds": prior_elapsed + time.monotonic() - session_start,
            }

        def save_checkpoint(name="checkpoint_last.pt"):
            nonlocal checkpoint_seconds, last_save
            started = time.monotonic()
            atomic_torch_save(root / name, snapshot())
            checkpoint_seconds = max(checkpoint_seconds, time.monotonic() - started)
            last_save = time.monotonic()
            LOG.info("Saved %s | step=%s | duration=%s", name, f"{completed:,}",
                     duration(time.monotonic() - started))

        def validation(writer, handle):
            nonlocal best_val, last_eval, last_val_step
            started = time.monotonic()
            values = evaluate(model, store, settings["eval_batch_size"], settings["eval_batches"], dtype)
            writer.writerow({
                "event": "validation", "step": completed, "tokens": completed * tokens_per_step,
                "session_seconds": time.monotonic() - session_start,
                "total_seconds": prior_elapsed + time.monotonic() - session_start,
                **{f"val_{key}": value for key, value in values.items()},
            })
            handle.flush()
            last_eval = time.monotonic()
            last_val_step = completed
            LOG.info("Validation | %s | duration=%s",
                     " | ".join(f"{k}={v:.4f}" for k, v in values.items()),
                     duration(last_eval - started))
            if values["weighted"] < best_val:
                best_val = values["weighted"]
                save_checkpoint("checkpoint_best.pt")
            return last_eval - started

        fields = ["event", "step", "tokens", "session_seconds", "total_seconds", "lr",
                  "train_loss", "grad_norm", "optimizer_step_skipped", "train_tokens_per_second",
                  "val_weighted"] + [f"val_{name}" for name in store.names]
        stopped = "completed"
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            eval_seconds = 0.0
            if not args.resume and not stop_request["reason"]:
                eval_seconds = validation(writer, handle)
            save_checkpoint()
            while completed < planned_steps:
                reserve = max(30.0, 2 * checkpoint_seconds)
                if stop_request["reason"]:
                    stopped = stop_request["reason"]
                    LOG.warning("Stop requested (%s); saving at step boundary", stopped)
                    break
                if time.monotonic() + smooth_seconds + reserve >= deadline:
                    stopped = "time_limit"
                    break
                step = completed
                lr = settings["lr"] * lr_factor(step, planned_steps, settings["warmup"], settings["decay"])
                started = time.monotonic()
                loss, grad_norm, skipped = update(lr)
                elapsed = time.monotonic() - started
                completed += 1
                train_seconds += elapsed
                session_train_tokens += tokens_per_step
                smooth_seconds = .95 * smooth_seconds + .05 * elapsed
                interval_losses.append(loss)
                writer.writerow({
                    "event": "train", "step": completed, "tokens": completed * tokens_per_step,
                    "session_seconds": time.monotonic() - session_start,
                    "total_seconds": prior_elapsed + time.monotonic() - session_start,
                    "lr": lr, "train_loss": loss, "grad_norm": grad_norm,
                    "optimizer_step_skipped": int(skipped),
                    "train_tokens_per_second": tokens_per_step / elapsed,
                })
                if skipped:
                    LOG.warning("FP16 overflow: optimizer update skipped at step %d", completed)
                if completed % settings["log_every"] == 0 or completed == planned_steps:
                    now = time.monotonic()
                    train_rate = session_train_tokens / max(train_seconds, 1e-9)
                    wall_rate = session_train_tokens / max(now - session_start, 1e-9)
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    eta = (planned_steps - completed) * smooth_seconds
                    phase = ("warmup" if step < max(1, round(planned_steps * settings["warmup"]))
                             else "decay" if step >= decay_start else "stable")
                    LOG.info(
                        "%s/%s (%.1f%%) | loss=%.4f avg=%.4f | lr=%.2e %s | "
                        "train=%s tok/s wall=%s tok/s | elapsed=%s ETA(train)=%s budget=%s | "
                        "VRAM=%.2f GiB peak=%.2f GiB",
                        f"{completed:,}", f"{planned_steps:,}", 100 * completed / planned_steps,
                        loss, sum(interval_losses) / len(interval_losses), lr, phase,
                        f"{train_rate:,.0f}", f"{wall_rate:,.0f}", duration(now - session_start),
                        duration(eta), duration(deadline - now), allocated, peak,
                    )
                    interval_losses.clear()
                    handle.flush()
                due_eval = time.monotonic() - last_eval >= settings["eval_every_minutes"] * 60
                if (due_eval or completed == planned_steps) and not stop_request["reason"]:
                    if time.monotonic() + max(eval_seconds, 10) + reserve < deadline:
                        eval_seconds = validation(writer, handle)
                if (time.monotonic() - last_save >= settings["checkpoint_every_minutes"] * 60
                        or completed == decay_start):
                    save_checkpoint()
            handle.flush()
            if (last_val_step != completed and not stop_request["reason"]
                    and time.monotonic() + max(eval_seconds, 10) + 30 < deadline):
                validation(writer, handle)
        save_checkpoint()
        optimizer.zero_grad(set_to_none=True)
        model.config.use_cache = True
        model.save_pretrained(root, safe_serialization=True)
        tok.save_pretrained(root)
        result = {
            "run_id": run_id, "session_id": session_id, "completed_steps": completed,
            "planned_steps": planned_steps, "training_tokens": completed * tokens_per_step,
            "session_training_tokens": (completed - initial_completed) * tokens_per_step,
            "stop_reason": stopped, "best_val": best_val if math.isfinite(best_val) else None,
            "session_elapsed_seconds": time.monotonic() - session_start,
            "total_elapsed_seconds": prior_elapsed + time.monotonic() - session_start,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "metrics_file": metrics_path.name,
        }
        write_json(root / "result.json", result)
        LOG.info("Finished: %s | step=%s/%s | elapsed=%s | export=%s",
                 stopped, f"{completed:,}", f"{planned_steps:,}",
                 duration(time.monotonic() - session_start), root)
    except Exception:
        LOG.exception("Training failed. Previous atomic checkpoints are retained; unsafe state is not exported.")
        raise
    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)


def sample(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    root = Path(args.out)
    tokenizer_path = root / "tokenizer" if (root / "tokenizer").exists() else root
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.is_absolute() and not path.exists():
            path = root / path
        state = load_checkpoint(path)
        config = LlamaConfig.from_dict(state["config"])
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config)
        model.load_state_dict(state["model"])
        del state
        model = model.to(device="cuda", dtype=dtype).eval()
        LOG.info("Sampling checkpoint: %s", path)
    else:
        model = LlamaForCausalLM.from_pretrained(root, dtype=dtype, attn_implementation="sdpa").cuda().eval()
    model.config.use_cache = True
    seed_all(args.seed)
    x = tok(args.prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    remaining = model.config.max_position_embeddings - x.input_ids.shape[1]
    if remaining <= 0 or x.input_ids.shape[1] == 0:
        raise ValueError("Prompt must be nonempty and shorter than the configured context.")
    options = {
        "max_new_tokens": min(args.new_tokens, remaining), "do_sample": not args.greedy,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tok.eos_token_id, "eos_token_id": tok.eos_token_id,
    }
    if not args.greedy:
        options.update(temperature=args.temperature, top_p=args.top_p)
    with torch.inference_mode():
        y = model.generate(**x, **options)
    print(tok.decode(y[0], skip_special_tokens=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare", help="Prepare a NEW token dataset directory")
    prep.add_argument("--data", default="mini_data")
    prep.add_argument("--train-tokens", type=int, default=1_000_000_000)
    prep.add_argument("--val-tokens", type=int, default=1_048_576, help="Per source")
    prep.add_argument("--code-jsonl")
    prep.add_argument("--seq-len", type=int, default=512)
    prep.add_argument("--shuffle-buffer", type=int, default=1000)
    prep.add_argument("--max-chars", type=int, default=40_000)
    prep.add_argument("--max-doc-tokens", type=int, default=8192)
    prep.add_argument("--seed", type=int, default=42)
    prep.set_defaults(func=prepare)
    ver = subs.add_parser("verify", help="Check all token IDs, lengths and checksums (CPU)")
    ver.add_argument("--data", required=True)
    ver.add_argument("--seq-len", type=int, default=512)
    ver.add_argument("--seed", type=int, default=42)
    ver.set_defaults(func=verify)
    tr = subs.add_parser("train", help="New training run, or resume an unfinished plan")
    tr.add_argument("--data")
    tr.add_argument("--out")
    tr.add_argument("--resume", help="Path to a checkpoint from THIS script")
    tr.add_argument("--hours", type=float, default=4.0, help="Wall-time budget for THIS invocation")
    for key in ("seq_len", "batch_size", "accumulation", "eval_batch_size", "eval_batches",
                "log_every", "seed", "max_steps"):
        tr.add_argument("--" + key.replace("_", "-"), type=int, default=None)
    for key in ("lr", "eval_every_minutes", "checkpoint_every_minutes", "warmup", "decay"):
        tr.add_argument("--" + key.replace("_", "-"), type=float, default=None)
    tr.add_argument("--gradient-checkpointing", action="store_true", default=None)
    tr.set_defaults(func=train)
    sm = subs.add_parser("sample")
    sm.add_argument("--out", required=True)
    sm.add_argument("--checkpoint", help="Optional checkpoint file, e.g. checkpoint_best.pt")
    sm.add_argument("--prompt", default="The solar system consists of")
    sm.add_argument("--new-tokens", type=int, default=60)
    sm.add_argument("--seed", type=int, default=42)
    sm.add_argument("--greedy", action="store_true")
    sm.add_argument("--temperature", type=float, default=.8)
    sm.add_argument("--top-p", type=float, default=.95)
    sm.add_argument("--repetition-penalty", type=float, default=1.0)
    sm.set_defaults(func=sample)
    for sub in (prep, ver, tr, sm):
        sub.add_argument("--log-file", help="Plain-text logfile; appended on resume")
        sub.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()
    positive = (
        "train_tokens", "val_tokens", "seq_len", "shuffle_buffer", "max_chars", "max_doc_tokens",
        "batch_size", "accumulation", "eval_batch_size", "eval_batches", "log_every", "hours",
        "eval_every_minutes", "checkpoint_every_minutes", "max_steps", "new_tokens", "lr",
        "temperature", "repetition_penalty",
    )
    for key in positive:
        value = getattr(args, key, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.command == "sample" and not 0 < args.top_p <= 1:
        parser.error("--top-p must lie in (0, 1]")
    if hasattr(args, "seed") and args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must lie in [0, 2**32)")
    log_file = args.log_file
    if not log_file:
        if args.command == "train":
            directory = Path(args.resume).resolve().parent if args.resume else Path(args.out or "mini_model")
            log_file = directory / "train.log"
        elif args.command in ("prepare", "verify"):
            log_file = Path(args.data) / f"{args.command}.log"
        else:
            log_file = Path(args.out) / "sample.log"
    configure_logging(log_file, args.log_level)
    try:
        args.func(args)
    except KeyboardInterrupt:
        LOG.warning("Interrupted outside a training step; inspect existing checkpoints/data.")
        raise SystemExit(130)
    except Exception:
        if args.command != "train":
            LOG.exception("Command failed")
        else:
            LOG.error("Command failed; see traceback below")
        raise


if __name__ == "__main__":
    main()