# Toy typed-decision model

Run a small open-source LLM on CPU, but let **your own decoder function**
decide what it may output. You give a prompt and a fixed list of allowed
answers (an enum). The decoder sets the probability of every other token to 0
and rescales the rest. Instead of text, you get back a typed decision:

```json
{"decision": "bullish", "confidence": 0.91,
 "probabilities": {"bullish": 0.87, "bearish": 0.04, "neutral": 0.09}}
```

This is a toy take on "System One" decision models such as TypeSafe's Jev.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build, smaller
pip install -r requirements.txt
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct`: about 1 GB, downloaded
automatically on first run, and a few prompts per second on a laptop CPU.
Other small options: `HuggingFaceTB/SmolLM2-360M-Instruct`,
`HuggingFaceTB/SmolLM2-135M-Instruct` (smallest), `Qwen/Qwen2.5-1.5B-Instruct`
(better answers, slower).

## Run on a GPU (Hugging Face Jobs)

The scorer needs the model's full next-token logits, so the hosted Inference
API won't do; you run the model yourself. `scripts/hf_job.sh` launches a
[Hugging Face Job](https://huggingface.co/docs/huggingface_hub/guides/jobs)
(needs a Pro, Team or Enterprise account) that downloads the pushed branch,
runs `pytest -q`, then runs the experiments with `--device cuda`:

```bash
pip install -U huggingface_hub && hf auth login        # once, on your machine
scripts/hf_job.sh                                       # t4-small, current branch
FLAVOR=l4x1 scripts/hf_job.sh --dtype bfloat16 --model Qwen/Qwen2.5-7B-Instruct
CHAT=1 RESULTS_REPO=<you>/veggie-results scripts/hf_job.sh   # + chat baseline, upload results/
```

The job's disk is deleted when it ends: every result CSV and JSONL file is
printed at the end of the log (`hf jobs logs <job-id>`), and `RESULTS_REPO` also uploads
`results/` to a private dataset repo. Only pushed commits are run.

Locally or elsewhere, pick the device and precision with `--device auto|cpu|cuda`
and `--dtype float32|bfloat16|float16`, or `device:` / `dtype:` in
`experiments.yaml`. `auto` uses the GPU when there is one. `float32` is fine
up to ~1.5B parameters; use `bfloat16` for larger models (on an L4/A10G/A100;
a T4 has no native bfloat16). Your decoder always gets float32 logits on the CPU.

## Run the experiments

```bash
python run_experiments.py experiments.yaml
python run_experiments.py experiments.yaml --decoder my_decoder:sharp_decoder --out results/sharp.csv
python run_experiments.py experiments.yaml --chat-completion
```

Edit `experiments.yaml` to add prompts and each experiment's `answers`. Each
run prints a table per experiment and writes two files to `results/`:

- `<name>.jsonl`: one decision per prompt, in the format shown above.
- `<name>.csv`: one row per prompt and answer, with these columns:

| column | meaning |
|---|---|
| `probability` | probability of the answer after your decoder (sums to 1 per prompt) |
| `unconstrained_prob` | probability the unrestricted model gives the whole answer |
| `decision` | the answer with the highest `probability` |
| `confidence` | sum of `unconstrained_prob` over the allowed answers. If it's low, the model didn't want to give any of them, so improve the prompt |
| `tokens` | how the answer was split into tokens |

To compare against unconstrained generation, use `--chat-completion`. It
generates a greedy response from the same prompt and system message without
applying the decoder or adding the answer list to the prompt. The output CSV
and JSONL contain the full `completion` and `matched_answer` (an exact match
against one of the configured answers, when present). Use `--max-new-tokens`
to change the generation limit; chat runs are saved with a `_chat` filename
suffix by default.

## Write your own decoder

Edit `my_decoder.py`:

```python
def decoder(logits, allowed_ids):
    # logits: [vocab] raw scores, allowed_ids: token ids that may be emitted
    probs = torch.zeros_like(logits)
    probs[allowed_ids] = torch.softmax(logits, -1)[allowed_ids]
    return probs / probs.sum()
```

The scorer checks that whatever you return is non-negative, zero outside
`allowed_ids` and sums to 1. Point the config at your decoder with
`decoder: my_decoder:your_function`.

## Answers with more than one token

Answers like `stagflation` are several tokens. In `mode: trie` (the default),
the allowed answers form a token tree. Your decoder runs at each branch point,
with the next tokens that can continue some allowed answer as `allowed_ids`.
An answer's probability is the product along its path. With single-token
answers this is one masked softmax.

If one answer is a token prefix of another (for example `buy` and `buy more`),
the tree can't tell where the shorter answer ends. Use `mode: sequence` for
that case: it scores each whole answer under the plain model and passes the
vector of answer log-probabilities to your decoder as the "logits".

## From Python

```python
from constrained_llm import ConstrainedScorer
s = ConstrainedScorer(system_prompt="Answer with one word.")
r = s.score("Is gold a risk-off asset?", ["Yes", "No"])
r.decision, r.confidence, r.probabilities   # or r.to_dict()
```

## Tests

`pytest -q` runs against a tiny randomly initialised model, so no download is needed.
