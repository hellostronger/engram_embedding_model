#!/usr/bin/env bash
# bootstrap_gb10.sh — one-shot setup + first real run on an NVIDIA GB10 / DGX Spark box.
#
# Verified platform facts (docs/COMPUTE.md):
#   GB10 = Blackwell sm_121, ARM64 aarch64, 128 GB unified LPDDR5x, ~273 GB/s.
#   * PyTorch must be built with CUDA >= 12.8 (sm_120 binaries are compatible with sm_121).
#   * Do NOT install flash-attn: no aarch64/sm_121 wheels; SDPA is as fast or faster.
#   * pin_memory is pointless (unified memory) — leave it off.
#
# Usage:
#   chmod +x bootstrap_gb10.sh && ./bootstrap_gb10.sh
set -euo pipefail

DOMAIN="${DOMAIN:-medical}"
PRESET="${PRESET:-medium}"
BACKBONE="${BACKBONE:-BAAI/bge-small-zh-v1.5}"
EPOCHS="${EPOCHS:-1}"
BS="${BS:-64}"

log() { printf '\033[1;36m[gb10]\033[0m %s\n' "$*"; }

# --------------------------------------------------------------------------- #
# 1. Environment
# --------------------------------------------------------------------------- #
log "python: $(python --version 2>&1)"
log "pip:    $(pip --version 2>&1 | head -1)"

python - <<'PY' || true
import torch
print(f"torch  = {torch.__version__}")
print(f"cuda   = {torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device = {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability()}")
    print(f"mem    = {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
PY

# Warn (do not abort) if torch lacks CUDA support — the code still runs on CPU.
python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("\033[1;33m[gb10] WARNING: no CUDA device found.\033[0m")
    print("  Training will fall back to CPU (slow but correct).")
    print("  On a GB10 box, install the NGC PyTorch container instead:")
    print("    docker run --gpus all -it nvcr.io/nvidia/pytorch:25.08-py3")
PY

# --------------------------------------------------------------------------- #
# 2. Dependencies
# --------------------------------------------------------------------------- #
log "installing dependencies"
pip install -q -r requirements.txt

# flash-attn is intentionally NOT installed (see header). Force SDPA in transformers.
export TRANSFORMERS_ATTN_BACKEND=sdpa
export TOKENIZERS_PARALLELISM=false

# --------------------------------------------------------------------------- #
# 3. Data
# --------------------------------------------------------------------------- #
mkdir -p data runs
if [ ! -f data/synth/train.jsonl ]; then
    log "generating synthetic seed data (domain=${DOMAIN})"
    python make_synthetic_data.py --out data/synth --domain "${DOMAIN}" --n 2000
else
    log "synthetic data already present, skipping"
fi

# Real data (needs network). Each dataset is small; failures are non-fatal.
log "preparing real datasets (best-effort)"
for DS in med_qa pubmed_qa case_hold; do
    if [ ! -f "data/${DS}.jsonl" ]; then
        python prepare_data.py --dataset "${DS}" --out "data/${DS}.jsonl" --limit 20000 \
            || log "  ${DS} failed (check network/gating) — continuing"
    fi
done

# --------------------------------------------------------------------------- #
# 4. Sanity check (offline, CPU)
# --------------------------------------------------------------------------- #
log "running offline self-check"
python offline_test.py

# --------------------------------------------------------------------------- #
# 5. First real run
# --------------------------------------------------------------------------- #
log "running first real experiment (preset=${PRESET}, backbone=${BACKBONE})"
python run_gb10.py \
    --preset "${PRESET}" \
    --backbone "${BACKBONE}" \
    --train data/synth/train.jsonl \
    --new-terms data/synth/cold_start.jsonl \
    --epochs "${EPOCHS}" \
    --batch-size "${BS}" \
    --stage all \
    --out "runs/de_${PRESET}"

log "done. results in runs/"
