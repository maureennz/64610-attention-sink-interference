# heads/ — retrieval heads and attention sinks (Person C)

Finds the retrieval heads and the sink position(s) for each model. Other parts of the project
should only need the JSON outputs in `results/<model>/`, never this code.

## Outputs (committed)

`results/<model>/retrieval_scores.json`
- `retrieval_heads`: heads with Wu et al. retrieval score ≥ `threshold` (0.1), sorted, as `{layer, head, score}`
- `top_50`: the 50 highest-scoring heads, in case the threshold leaves too few
- `scores`: full `[layer][head]` matrix
- `niah_exact_match` / `niah_token_accuracy`: sanity check that the model actually solves the needle task

`results/<model>/sinks.json`
- `sink_positions`: token positions that act as sinks (expected: `[0]`)
- `prepended_special_tokens`: what the tokenizer puts at the start (Qwen2.5: nothing, Llama 3: `<|begin_of_text|>`).
  **Use the same convention when building prompts**, or the sink may land somewhere else.
- `per_head_sink_mass`: `[layer][head]` mean attention mass on each sink position (baseline for interventions)
- `content_sink_candidates`: tokens that also attract sink-like attention (e.g. newlines), if any

## Methods

**Retrieval score** (Wu et al. 2024): needle-in-a-haystack over WikiText filler, 3 context lengths
(500/1000/2000) × 11 depths × 4 reps. A head copy-pastes an answer token when the model predicts it
correctly, the head's top-1 attention is inside the needle value, and the attended token is the one
being produced. Score = fraction of answer tokens copy-pasted, averaged over samples. The answer is
teacher-forced instead of generated, which matches the generation-time check on every correct step.

**Sinks (median-mass)**: on 64 random WikiText chunks of 1024 tokens, compute each head's mean
attention from late queries (second half) to each early key (first half); take the median across all
heads per position; flag positions above τ = 0.1. A position is a sink if it is flagged in ≥ 90% of
chunks. Because the chunks start at random places, a sink at position 0 is positional, not tied to a
specific token.

Loading uses TransformerLens with no weight processing, so attention scores equal the HF model's.

## Running

```bash
module load miniforge/25.11.0-0 && conda activate attn-sink
pip install -r heads/requirements.txt
sbatch heads/run_heads.sbatch qwen2.5-1.5b   # from the repo root; log goes to slurm-heads-<jobid>.out
```

Models are loaded by Hugging Face ID with a pinned revision (`common.py`), so set `HF_HOME` if your
cache isn't in the default location. Llama-3.2-1B is gated: request access on its HF page and run
`huggingface-cli login`.
