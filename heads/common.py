"""Shared helpers for the heads/ analysis: model loading, tokenization, filler text, paths."""
import json
import random
import re
from pathlib import Path

import torch
from transformer_lens.model_bridge import TransformerBridge

# Short key -> (Hugging Face ID, pinned revision). Pinning keeps everyone on identical weights.
MODELS = {
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
    "llama-3.2-1b": ("meta-llama/Llama-3.2-1B", None),  # TODO: pin revision when we start the week-4 port
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_model(key, device=None, dtype=torch.float32):
    """Load a pinned model as a TransformerLens 4.x TransformerBridge.

    The bridge keeps raw HF weights and eager attention, so attention scores match the HF model.
    Compatibility mode (without weight processing) only adds the legacy hook names, e.g.
    blocks.{l}.attn.hook_pattern.
    """
    hf_name, revision = MODELS[key]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerBridge.boot_transformers(hf_name, revision=revision, device=device, dtype=dtype)
    model.enable_compatibility_mode(no_processing=True, disable_warnings=True)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def bos_ids(tokenizer):
    """Special tokens the HF tokenizer prepends by default: [] for Qwen2.5, [<|begin_of_text|>] for Llama 3.

    We mimic HF defaults rather than TransformerLens's prepend_bos, because which token sits at
    position 0 is exactly what determines where the sink lands.
    """
    return tokenizer("", add_special_tokens=True)["input_ids"]


def encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def load_filler_sentences():
    """WikiText-2 (test split) sentences, lightly cleaned, for haystack / neutral text."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    lines = [l.strip() for l in ds["text"] if l.strip() and not l.strip().startswith("=")]
    text = " ".join(lines)
    for a, b in [(" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" ,", ","), (" .", "."), (" 's", "'s")]:
        text = text.replace(a, b)
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]


def filler_token_chunks(tokenizer, sentences, n_tokens, rng: random.Random):
    """Token lists for consecutive sentences from a random start, totalling exactly n_tokens."""
    start = rng.randrange(len(sentences))
    chunks, total, i = [], 0, start
    while total < n_tokens:
        ids = encode(tokenizer, " " + sentences[i % len(sentences)])
        ids = ids[: n_tokens - total]
        chunks.append(ids)
        total += len(ids)
        i += 1
    return chunks


def model_results_dir(key):
    d = RESULTS_DIR / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"wrote {path}")
