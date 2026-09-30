"""Run constrained-word experiments from a YAML file and save the probabilities.

Usage:
    python run_experiments.py experiments.yaml
    python run_experiments.py experiments.yaml --out results/run1.csv --decoder my_decoder:sharp_decoder
    python run_experiments.py experiments.yaml --device cuda --dtype bfloat16
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from constrained_llm import DEFAULT_MODEL, ConstrainedScorer


def load_decoder(spec: str):
    module_name, _, func_name = spec.partition(":")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module_name), func_name or "decoder")


DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_device(spec: str) -> str:
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {spec} requested but no CUDA GPU is available")
    return spec


def run(config: dict, scorer: ConstrainedScorer) -> tuple[pd.DataFrame, list[dict]]:
    """Score every prompt; return a long table and one Jev-style decision per prompt."""
    records, decisions = [], []
    for exp in config["experiments"]:
        name = exp["name"]
        # "words" is the older name for "answers".
        answers = [str(a) for a in exp.get("answers", exp.get("words", []))]
        for k, prompt in enumerate(exp["prompts"]):
            r = scorer.score(
                prompt,
                answers,
                system_prompt=exp.get("system_prompt"),
                leading_space=exp.get("leading_space"),
            )
            decisions.append({"experiment": name, "prompt_id": k, **r.to_dict()})
            for a in answers:
                records.append(
                    {
                        "experiment": name,
                        "prompt_id": k,
                        "prompt": prompt,
                        "answer": a,
                        "probability": r.probabilities[a],
                        "unconstrained_prob": r.unconstrained_probs[a],
                        "decision": r.decision,
                        "confidence": r.confidence,
                        "tokens": "|".join(r.tokens[a]),
                    }
                )
    return pd.DataFrame.from_records(records), decisions


def run_chat_completions(
    config: dict, scorer: ConstrainedScorer, max_new_tokens: int
) -> tuple[pd.DataFrame, list[dict]]:
    """Generate an unrestricted chat completion for each configured prompt."""
    records = []
    for exp in config["experiments"]:
        name = exp["name"]
        answers = [str(a) for a in exp.get("answers", exp.get("words", []))]
        for prompt_id, prompt in enumerate(exp["prompts"]):
            completion = scorer.chat_complete(
                prompt,
                system_prompt=exp.get("system_prompt"),
                max_new_tokens=max_new_tokens,
            )
            normalized = completion.strip().casefold().rstrip(".!?")
            matched_answer = next((a for a in answers if a.casefold() == normalized), None)
            records.append(
                {
                    "experiment": name,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "answers": "|".join(answers),
                    "completion": completion,
                    "matched_answer": matched_answer,
                }
            )
    df = pd.DataFrame.from_records(records)
    return df, records


def print_report(df: pd.DataFrame) -> None:
    for name, g in df.groupby("experiment", sort=False):
        wide = g.pivot_table(index="prompt_id", columns="answer", values="probability", sort=False)
        meta = g.groupby("prompt_id").agg(decision=("decision", "first"), confidence=("confidence", "first"),
                                          prompt=("prompt", "first"))
        meta["prompt"] = meta["prompt"].str.slice(0, 60)
        print(f"\n=== {name} ===")
        print(wide.join(meta).to_string(float_format=lambda x: f"{x:.4f}"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="experiments.yaml")
    ap.add_argument("--out", help="CSV output path (default results/<config>_<timestamp>.csv)")
    ap.add_argument("--model", help="override the model in the config")
    ap.add_argument("--decoder", help="override the decoder, as module:function")
    ap.add_argument("--mode", choices=["trie", "sequence"], help="override the scoring mode")
    ap.add_argument("--device", help="cpu, cuda, cuda:1 or auto (default: config, else auto)")
    ap.add_argument("--dtype", choices=list(DTYPES), help="model weights dtype (default: config, else float32)")
    ap.add_argument(
        "--chat-completion",
        action="store_true",
        help="generate unrestricted greedy responses instead of constrained scores",
    )
    ap.add_argument("--max-new-tokens", type=int, default=32, help="chat completion length limit")
    args = ap.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    model = args.model or config.get("model", DEFAULT_MODEL)
    decoder_spec = args.decoder or config.get("decoder", "my_decoder:decoder")
    device = resolve_device(args.device or config.get("device", "auto"))
    dtype = args.dtype or config.get("dtype", "float32")
    if dtype not in DTYPES:
        raise SystemExit(f"dtype must be one of {list(DTYPES)}, got {dtype!r}")

    print(f"Loading {model} on {device} ({dtype}) ...", flush=True)
    t0 = time.time()
    scorer = ConstrainedScorer(
        model,
        decoder=load_decoder(decoder_spec),
        use_chat_template=config.get("use_chat_template"),
        system_prompt=config.get("system_prompt"),
        mode=args.mode or config.get("mode", "trie"),
        device=device,
        dtype=DTYPES[dtype],
    )
    print(f"Loaded in {time.time() - t0:.1f}s; decoder={decoder_spec}, mode={scorer.mode}")

    t0 = time.time()
    if args.chat_completion:
        df, decisions = run_chat_completions(config, scorer, args.max_new_tokens)
        print(df[["experiment", "prompt_id", "completion", "matched_answer"]].to_string(index=False))
    else:
        df, decisions = run(config, scorer)
        print_report(df)

    suffix = "_chat" if args.chat_completion else ""
    out = Path(
        args.out
        or f"results/{Path(args.config).stem}{suffix}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    jsonl = out.with_suffix(".jsonl")
    jsonl.write_text("".join(json.dumps(d) + "\n" for d in decisions))
    print(f"\n{len(decisions)} prompts in {time.time() - t0:.1f}s. Saved {out} and {jsonl}")


if __name__ == "__main__":
    main()
