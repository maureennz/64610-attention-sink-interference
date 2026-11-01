"""Retrieval scores (Wu et al., "Retrieval Head Mechanistically Explains Long-Context Factuality").

Needle-in-a-haystack: a needle sentence containing a random multi-token value is inserted into
WikiText filler at varying context lengths and depths, and the model is asked to recall the value.

Head h "copy-pastes" answer token w at a decoding step if
  (1) the model's prediction at that step is correct (w),
  (2) h's top-1 attended position j lies inside the needle value's span, and
  (3) the token at j equals w.
Retrieval score of h on a sample = (# answer tokens h copy-pastes) / (# answer tokens);
the reported score is the mean over samples.

We teacher-force the answer in one forward pass instead of generating. For every step where the
model's greedy prediction is correct this is identical to Wu et al.'s generation-time check, and
condition (1) excludes the rest.

Usage: python heads/retrieval_score.py --model qwen2.5-1.5b
"""
import argparse
import random

import numpy as np
import torch

from common import (MODELS, bos_ids, encode, filler_token_chunks, load_filler_sentences, load_model,
                    model_results_dir, save_json)

# (needle sentence, question, answer prefix). The value is random, so it can't come from prior knowledge.
NEEDLES = [
    ("The secret passphrase for the {key} is {value}.",
     "What is the secret passphrase for the {key}?", "The secret passphrase for the {key} is"),
    ("The special magic code hidden in the {key} is {value}.",
     "What is the special magic code hidden in the {key}?", "The special magic code hidden in the {key} is"),
    ("Remember this: the password to open the {key} is {value}.",
     "What is the password to open the {key}?", "The password to open the {key} is"),
    ("The best thing to say at the {key} is {value}.",
     "What is the best thing to say at the {key}?", "The best thing to say at the {key} is"),
]
KEYS = ["old lighthouse", "harbor warehouse", "mountain cabin", "museum basement", "garden shed",
        "city archive", "northern bridge", "train station locker", "university library", "desert outpost"]
WORDS = ["violet", "lantern", "copper", "meadow", "falcon", "orbit", "pebble", "thunder", "saffron",
         "glacier", "velvet", "compass", "ember", "willow", "quartz", "harbor", "cobalt", "juniper",
         "marble", "nectar", "prism", "tundra", "walnut", "zephyr"]


def random_value(rng):
    if rng.random() < 0.5:
        return " ".join(rng.sample(WORDS, 3))
    return str(rng.randrange(100000, 1000000))


def build_sample(tok, sentences, context_len, depth, rng):
    """Token ids for haystack+needle+question, the value ids, and the value's span in the haystack."""
    needle_t, question_t, answer_t = rng.choice(NEEDLES)
    key, value = rng.choice(KEYS), random_value(rng)
    pre, post = needle_t.split("{value}")
    needle_pre = encode(tok, " " + pre.format(key=key).rstrip())
    value_ids = encode(tok, " " + value)  # same ids in needle and answer, so copy targets match exactly
    needle_post = encode(tok, post)
    question = encode(tok, f"\n\nQuestion: {question_t.format(key=key)}\nAnswer: {answer_t.format(key=key)}")
    bos = bos_ids(tok)

    budget = context_len - len(bos) - len(needle_pre) - len(value_ids) - len(needle_post) - len(question)
    chunks = filler_token_chunks(tok, sentences, budget, rng)
    # Insert at the sentence boundary closest to the requested depth.
    bounds = np.cumsum([0] + [len(c) for c in chunks])
    cut = int(np.argmin(np.abs(bounds - depth * budget)))
    before = [t for c in chunks[:cut] for t in c]
    after = [t for c in chunks[cut:] for t in c]

    prefix = bos + before + needle_pre
    span = (len(prefix), len(prefix) + len(value_ids))
    ids = prefix + value_ids + needle_post + after + question
    return {"input_ids": ids, "value_ids": value_ids, "span": span, "depth": depth,
            "context_len": len(ids)}


def copy_paste(model, sample):
    """Returns ([layers, heads] copy counts, n answer tokens, per-step correctness)."""
    ids = sample["input_ids"] + sample["value_ids"]
    tokens = torch.tensor([ids], device=model.cfg.device)
    n, P = len(sample["value_ids"]), len(sample["input_ids"])
    q_pos = torch.arange(P - 1, P - 1 + n, device=tokens.device)  # positions predicting each value token
    argmax = {}

    def hook(pattern, hook):
        argmax[hook.layer()] = pattern[0, :, q_pos, :].argmax(dim=-1)  # [heads, n]

    logits = model.run_with_hooks(tokens, fwd_hooks=[(lambda nm: nm.endswith("hook_pattern"), hook)])
    targets = tokens[0, q_pos + 1]
    correct = logits[0, q_pos].argmax(dim=-1) == targets  # [n]
    j = torch.stack([argmax[l] for l in range(model.cfg.n_layers)])  # [L, H, n]
    s, e = sample["span"]
    copies = (j >= s) & (j < e) & (tokens[0][j] == targets) & correct
    return copies.sum(dim=-1).cpu().numpy(), n, correct.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-1.5b", choices=MODELS)
    p.add_argument("--context-lens", type=int, nargs="+", default=[500, 1000, 2000])
    p.add_argument("--n-depths", type=int, default=11)
    p.add_argument("--reps", type=int, default=4, help="samples per (length, depth) cell")
    p.add_argument("--threshold", type=float, default=0.1, help="Wu et al. use 0.1")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    model = load_model(args.model)
    tok = model.tokenizer
    sentences = load_filler_sentences()

    per_sample_scores, token_acc, exact = [], [], []
    for L in args.context_lens:
        for depth in np.linspace(0, 1, args.n_depths):
            for _ in range(args.reps):
                sample = build_sample(tok, sentences, L, float(depth), rng)
                copies, n, correct = copy_paste(model, sample)
                per_sample_scores.append(copies / n)
                token_acc.append(correct.mean())
                exact.append(bool(correct.all()))

    scores = np.mean(per_sample_scores, axis=0)  # [L, H]
    order = np.argsort(-scores, axis=None)
    ranked = [{"layer": int(l), "head": int(h), "score": round(float(scores[l, h]), 4)}
              for l, h in zip(*np.unravel_index(order, scores.shape))]

    hf_name, revision = MODELS[args.model]
    out = {
        "model": hf_name,
        "revision": revision,
        "n_samples": len(per_sample_scores),
        "context_lens": args.context_lens,
        "n_depths": args.n_depths,
        "niah_token_accuracy": round(float(np.mean(token_acc)), 4),
        "niah_exact_match": round(float(np.mean(exact)), 4),
        "threshold": args.threshold,
        "retrieval_heads": [r for r in ranked if r["score"] >= args.threshold],
        "top_50": ranked[:50],
        "scores": scores.round(4).tolist(),  # [layer][head]
    }
    save_json(out, model_results_dir(args.model) / "retrieval_scores.json")
    print(f"NIAH exact match {out['niah_exact_match']:.2f}, token acc {out['niah_token_accuracy']:.2f}")
    print(f"{len(out['retrieval_heads'])} heads with score >= {args.threshold}; top 10:")
    for r in ranked[:10]:
        print(f"  L{r['layer']}H{r['head']}: {r['score']:.3f}")


if __name__ == "__main__":
    main()
