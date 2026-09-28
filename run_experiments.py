"""Run constrained-word experiments from a YAML file and save the probabilities.

Usage:
    python run_experiments.py experiments.yaml
    python run_experiments.py experiments.yaml --out results/run1.csv --decoder my_decoder:sharp_decoder
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from constrained_llm import DEFAULT_MODEL, ConstrainedScorer


def load_decoder(spec: str):
    module_name, _, func_name = spec.partition(":")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module_name), func_name or "decoder")


def run(config: dict, scorer: ConstrainedScorer) -> pd.DataFrame:
    records = []
    for exp in config["experiments"]:
        name, words = exp["name"], [str(w) for w in exp["words"]]
        for k, prompt in enumerate(exp["prompts"]):
            r = scorer.score(
                prompt,
                words,
                system_prompt=exp.get("system_prompt"),
                leading_space=exp.get("leading_space"),
            )
            for w in words:
                records.append(
                    {
                        "experiment": name,
                        "prompt_id": k,
                        "prompt": prompt,
                        "word": w,
                        "prob": r.probs[w],
                        "unconstrained_prob": r.unconstrained_probs[w],
                        "coverage": r.coverage,
                        "top_word": r.top_word,
                        "tokens": "|".join(r.tokens[w]),
                    }
                )
    return pd.DataFrame.from_records(records)


def print_report(df: pd.DataFrame) -> None:
    for name, g in df.groupby("experiment", sort=False):
        wide = g.pivot_table(index="prompt_id", columns="word", values="prob", sort=False)
        meta = g.groupby("prompt_id").agg(coverage=("coverage", "first"), top=("top_word", "first"),
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
    args = ap.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    model = args.model or config.get("model", DEFAULT_MODEL)
    decoder_spec = args.decoder or config.get("decoder", "my_decoder:decoder")

    print(f"Loading {model} ...", flush=True)
    t0 = time.time()
    scorer = ConstrainedScorer(
        model,
        decoder=load_decoder(decoder_spec),
        use_chat_template=config.get("use_chat_template"),
        system_prompt=config.get("system_prompt"),
        mode=args.mode or config.get("mode", "trie"),
    )
    print(f"Loaded in {time.time() - t0:.1f}s; decoder={decoder_spec}, mode={scorer.mode}")

    t0 = time.time()
    df = run(config, scorer)
    print_report(df)

    out = Path(args.out or f"results/{Path(args.config).stem}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{df['prompt_id'].groupby(df['experiment']).nunique().sum()} prompts in "
          f"{time.time() - t0:.1f}s. Saved {out}")


if __name__ == "__main__":
    main()
