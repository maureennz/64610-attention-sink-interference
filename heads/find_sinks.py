"""Locate each model's attention sink(s) empirically with the median-mass approach.

For each WikiText chunk (random start, so the first token varies):
  1. For every head, take the attention that late queries (second half of the sequence) send
     to each early key position (first half). Every early key is visible to the same set of
     queries, so masses are comparable across positions and free of recency effects.
  2. Take the median of that mass across all heads (layers x heads) for each key position.
  3. Flag a position as a sink in that chunk if the median mass exceeds --tau.

A position is reported as a sink if it is flagged in at least --min-rate of chunks. Tokens that
get flagged away from fixed positions (e.g. newlines, punctuation) are reported separately as
content-dependent sink candidates.

Usage: python heads/find_sinks.py --model qwen2.5-1.5b
"""
import argparse
import random
from collections import Counter

import numpy as np
import torch

from common import (MODELS, bos_ids, filler_token_chunks, load_filler_sentences, load_model,
                    model_results_dir, save_json)


def received_mass(model, tokens):
    """[n_layers, n_heads, seq//2]: mean attention from late queries to each early key."""
    half = tokens.shape[1] // 2
    out = {}

    def hook(pattern, hook):
        out[hook.layer()] = pattern[0, :, half:, :half].mean(dim=1).float().cpu()

    model.run_with_hooks(tokens, return_type=None,
                         fwd_hooks=[(lambda n: n.endswith("hook_pattern"), hook)])
    return torch.stack([out[l] for l in range(model.cfg.n_layers)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-1.5b", choices=MODELS)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--tau", type=float, default=0.1, help="median-mass threshold for flagging")
    p.add_argument("--min-rate", type=float, default=0.9, help="flag rate needed to call a position a sink")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    model = load_model(args.model)
    tok = model.tokenizer
    sentences = load_filler_sentences()
    bos = bos_ids(tok)

    half = args.seq_len // 2
    flag_counts = np.zeros(half)
    median_sum = np.zeros(half)
    mass_sum = None  # [layers, heads, half]
    flagged_tokens = Counter()  # tokens flagged at positions that are not fixed sinks (filled in below)
    flagged_records = []

    for _ in range(args.n_samples):
        chunks = filler_token_chunks(tok, sentences, args.seq_len - len(bos), rng)
        ids = bos + [t for c in chunks for t in c]
        tokens = torch.tensor([ids], device=model.cfg.device)
        mass = received_mass(model, tokens)  # [L, H, half]
        med = mass.flatten(0, 1).median(dim=0).values.numpy()  # [half]
        flags = med > args.tau
        flag_counts += flags
        median_sum += med
        mass_sum = mass.numpy() if mass_sum is None else mass_sum + mass.numpy()
        flagged_records.append([(int(k), ids[k]) for k in np.nonzero(flags)[0]])

    rates = flag_counts / args.n_samples
    sink_positions = [int(k) for k in np.nonzero(rates >= args.min_rate)[0]]
    for rec in flagged_records:
        for k, t in rec:
            if k not in sink_positions:
                flagged_tokens[tok.decode([t])] += 1

    mean_mass = mass_sum / args.n_samples  # [L, H, half]
    per_head = {str(k): mean_mass[:, :, k].round(4).tolist() for k in sink_positions}

    hf_name, revision = MODELS[args.model]
    out = {
        "model": hf_name,
        "revision": revision,
        "method": "median over all heads of mean attention from late queries (second half) to each "
                  "early key (first half); flagged if > tau; sink if flagged in >= min_rate of samples",
        "n_samples": args.n_samples,
        "seq_len": args.seq_len,
        "tau": args.tau,
        "min_rate": args.min_rate,
        "prepended_special_tokens": [tok.decode([t]) for t in bos],
        "sink_positions": sink_positions,
        "flag_rate_first_16": rates[:16].round(3).tolist(),
        "median_mass_first_16": (median_sum[:16] / args.n_samples).round(4).tolist(),
        "other_flagged_positions": {int(k): round(float(r), 3) for k, r in enumerate(rates)
                                    if r > 0 and k not in sink_positions},
        "content_sink_candidates": flagged_tokens.most_common(20),
        "per_head_sink_mass": per_head,  # sink position -> [layer][head] mean mass
    }
    save_json(out, model_results_dir(args.model) / "sinks.json")
    print("sink positions:", sink_positions)
    print("median mass, first 8 positions:", out["median_mass_first_16"][:8])
    print("content sink candidates:", out["content_sink_candidates"][:5])


if __name__ == "__main__":
    main()
