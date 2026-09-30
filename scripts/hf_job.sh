#!/usr/bin/env bash
# Run the tests and experiments on a Hugging Face Jobs GPU. The job's disk is
# deleted when it ends, so the result CSV and JSONL files are printed to the log.
#
# Prerequisites (on your machine): pip install -U huggingface_hub && hf auth login
# HF Jobs needs a Pro, Team or Enterprise account.
#
# Usage:
#   scripts/hf_job.sh                                   # current branch, t4-small
#   FLAVOR=l4x1 scripts/hf_job.sh --dtype bfloat16      # extra args go to run_experiments.py
#   RESULTS_REPO=<user>/veggie-results scripts/hf_job.sh   # also upload results/ to a dataset repo
#
# Environment variables:
#   FLAVOR        GPU flavor (default t4-small; see `hf jobs hardware`)
#   BRANCH        git branch to run (default: the current branch; it must be pushed)
#   CONFIG        experiments file (default experiments.yaml)
#   REPO          GitHub repo (default githubgir/veggie; must be public)
#   IMAGE         Docker image with CUDA torch (default pytorch/pytorch:latest)
#   TIMEOUT       job timeout (default 1h)
#   RESULTS_REPO  optional HF dataset repo for results/; sends your HF_TOKEN as a job secret
#   CHAT=1        also run the unconstrained --chat-completion baseline
set -euo pipefail

FLAVOR=${FLAVOR:-t4-small}
BRANCH=${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}
CONFIG=${CONFIG:-experiments.yaml}
REPO=${REPO:-githubgir/veggie}
IMAGE=${IMAGE:-pytorch/pytorch:latest}
TIMEOUT=${TIMEOUT:-1h}
EXTRA_ARGS="$*"

if [[ -n "$(git status --porcelain 2>/dev/null)" ]] || \
   [[ -n "$(git log "origin/$BRANCH..HEAD" --oneline 2>/dev/null)" ]]; then
  echo "warning: the job runs origin/$BRANCH from GitHub, not your uncommitted or unpushed changes" >&2
fi

# Runs inside the container. The image has CUDA torch but no git, so fetch a tarball.
read -r -d '' REMOTE <<'EOF' || true
set -euo pipefail
python -c "import io, tarfile, urllib.request; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen('https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz').read())).extractall('src')"
cd src/*/
nvidia-smi --query-gpu=name,memory.total --format=csv
pip install -q -r requirements.txt
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
pytest -q
python run_experiments.py "$CONFIG" --device cuda $EXTRA_ARGS
if [[ "${CHAT:-0}" == "1" ]]; then
  python run_experiments.py "$CONFIG" --device cuda --chat-completion $EXTRA_ARGS
fi
for f in results/*.csv; do echo "===== $f ====="; cat "$f"; done
for f in results/*.jsonl; do echo "===== $f ====="; cat "$f"; done
if [[ -n "${RESULTS_REPO:-}" ]]; then
  pip install -q -U huggingface_hub
  hf repo create "$RESULTS_REPO" --repo-type dataset --private --exist-ok || true
  hf upload "$RESULTS_REPO" results "runs/$(date -u +%Y%m%d_%H%M%S)" --repo-type dataset
fi
EOF

args=(--flavor "$FLAVOR" --timeout "$TIMEOUT"
      -e REPO="$REPO" -e BRANCH="$BRANCH" -e CONFIG="$CONFIG"
      -e EXTRA_ARGS="$EXTRA_ARGS" -e CHAT="${CHAT:-0}")
if [[ -n "${RESULTS_REPO:-}" ]]; then
  args+=(-e RESULTS_REPO="$RESULTS_REPO" --secrets HF_TOKEN)
fi

echo "Launching $REPO@$BRANCH on $FLAVOR ..." >&2
exec hf jobs run "${args[@]}" "$IMAGE" bash -c "$REMOTE"
