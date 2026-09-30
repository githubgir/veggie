# AGENTS.md

Context for anyone, human or AI agent, picking up this project.

## What we're building

A **toy typed-decision model**, a small-scale take on "System One" models
such as TypeSafe's Jev. These return a typed value with probabilities
instead of free text.

1. Take a small open-source LLM that runs on a laptop CPU.
2. Replace its decoding step with **our own decoder function**. We never
   generate text; the LLM only supplies next-token logits.
3. For each prompt, allow only a handful of answers (an enum, e.g.
   `bullish / bearish / neutral`). The decoder sets the probability of every
   other token to 0 and rescales the allowed ones to sum to 1.
4. Return a typed decision per prompt:
   ```json
   {"decision": "bullish", "confidence": 0.91,
    "probabilities": {"bullish": 0.87, "bearish": 0.04, "neutral": 0.09}}
   ```
5. Experiment with different decoders, prompts and answer sets, driven by
   an editable test file.

The owner works on quant models: signals/alpha, portfolio construction,
asset allocation, analytics. Expect finance-flavoured prompts such as
headline sentiment or macro regimes.

## Decisions already made

| Topic | Decision |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (about 1 GB, CPU); swappable in the config |
| Answer type | Enum only: pick one from a fixed list. No bool/int/multi-label yet |
| Confidence | Coverage: the unrestricted model's total probability on the allowed answers |
| Input | Hand-edited `experiments.yaml`. No CSV/panel input for now |
| Prompt style | Chat template of the instruct model, with a system prompt |
| Default decoder | Mask to the allowed tokens, then renormalize (`my_decoder.decoder`) |

## How it works

- `constrained_llm/scorer.py`: `ConstrainedScorer.score(prompt, answers)`
  returns a `ScoreResult` with `decision`, `confidence`, `probabilities`,
  `unconstrained_probs`, `tokens` and `to_dict()`.
- Multi-token answers (`mode: trie`, the default): the answers form a token
  trie. The decoder runs at each branch point with the node's children as
  `allowed_ids`, and an answer's probability is the product along its path.
  An answer that is a token prefix of another raises an error; use
  `mode: sequence` for those, which renormalizes whole-answer log-probs.
- All answers for a prompt are scored in one right-padded forward pass.
- The decoder signature is
  `decoder(logits[V], allowed_ids[k]) -> probs[V]`. The scorer checks that the
  output is non-negative, zero outside `allowed_ids` and sums to 1.
- `run_experiments.py` reads `experiments.yaml` and prints a table per
  experiment. It writes `results/<name>.csv` (one row per prompt and answer)
  and `results/<name>.jsonl` (one decision per prompt).

## Files

| File | Purpose |
|---|---|
| `experiments.yaml` | **Edit this**: model, decoder, system prompt; experiments with `answers` and `prompts` |
| `my_decoder.py` | **Edit this**: your decoder function(s) |
| `constrained_llm/` | Scorer and built-in decoders |
| `run_experiments.py` | Command-line runner (`--device`, `--dtype` for GPU) |
| `scripts/hf_job.sh` | Runs tests + experiments on a Hugging Face Jobs GPU |
| `tests/test_scorer.py` | Tests on a tiny random GPT-2, so no download is needed |

## Run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q
python run_experiments.py experiments.yaml
python run_experiments.py experiments.yaml --decoder my_decoder:sharp_decoder
```

## Current status and open items

- The first version passed its 9 tests on the tiny model, and the
  command-line runner worked end to end.
- **The latest change (Jev-style `decision` / `confidence` /
  `probabilities`) was pushed without running the tests**, because the shell
  was unavailable. Run `pytest -q` first.
- **It has never been run on a real model.** The cloud sandbox blocked
  huggingface.co. In Claude Code on the web, allow `huggingface.co`,
  `*.huggingface.co` and `*.hf.co` in the environment's network settings,
  run it locally, or run `scripts/hf_job.sh` on a Hugging Face Jobs GPU.
- The `--device` / `--dtype` change and `scripts/hf_job.sh` were also pushed
  untested (the sandbox also blocks PyPI, so no torch). The HF job runs
  `pytest -q` first.
- Possible next steps: look at real Qwen results and tune the prompts where
  confidence is low; try decoders with priors/weights or temperature; later,
  CSV input to produce a date × ticker probability panel for signals.
