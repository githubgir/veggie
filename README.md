# Constrained-vocabulary LLM scoring

Run a small open-source LLM on CPU, but let **your own decoder function**
decide what it may output. You give a prompt and a short list of allowed
words. The decoder sets the probability of every other token to 0 and rescales
the rest, and you get back a probability for each allowed word.

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

## Run the experiments

```bash
python run_experiments.py experiments.yaml
python run_experiments.py experiments.yaml --decoder my_decoder:sharp_decoder --out results/sharp.csv
```

Edit `experiments.yaml` to add prompts and allowed words. Each run prints a
table per experiment and writes a long-format CSV to `results/` with these columns:

| column | meaning |
|---|---|
| `prob` | probability of the word after your decoder (sums to 1 per prompt) |
| `unconstrained_prob` | probability the unrestricted model gives the whole word |
| `coverage` | sum of `unconstrained_prob` over the allowed words. If it's low, the model didn't want to answer with any of them, so improve the prompt |
| `top_word` | the allowed word with the highest `prob` |
| `tokens` | how the word was split into tokens |

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

## Words with more than one token

Words like `stagflation` are several tokens. In `mode: trie` (the default), the
allowed words form a token tree. Your decoder runs at each branch point, with
the next tokens that can continue some allowed word as `allowed_ids`. A word's
probability is the product along its path. With single-token words this is one
masked softmax.

If one word is a token prefix of another (for example `buy` and `buy more`),
the tree can't tell where the shorter word ends. Use `mode: sequence` for that
case: it scores each whole word under the plain model and passes the vector
of word log-probabilities to your decoder as the "logits".

## From Python

```python
from constrained_llm import ConstrainedScorer
s = ConstrainedScorer(system_prompt="Answer with one word.")
r = s.score("Is gold a risk-off asset?", ["Yes", "No"])
r.probs, r.coverage
```

## Tests

`pytest -q` runs against a tiny randomly initialised model, so no download is needed.
