#!/usr/bin/env bash
# Smoke-test the project environment on counting@192.168.6.200 (lab server).
# Run from the repo root with the conda env activated.
# Usage: conda activate cs338 && bash scripts/verify_env.sh
# Exit code: 0 if ALL checks pass, 1 otherwise.
#
# Data is read from NAS: /mnt/mmlab2024nas/counting (override via DATA_BASE env var)

set -uo pipefail

PASS=0
FAIL=0
NAS_DATA="${DATA_BASE:-/mnt/mmlab2024nas/counting}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

check_pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
check_fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=============================================="
echo " VA-Count Environment Verification"
echo " Server: counting@192.168.6.200"
echo " Data  : ${NAS_DATA}"
echo "=============================================="

# --- 1. Python import checks ---
echo ""
echo "[1/5] Python import checks..."

if python -c "import torch; print(f'  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
    check_pass "torch"
else
    check_fail "torch (import failed)"
fi

if python -c "import clip; print('  clip OK')" 2>/dev/null; then
    check_pass "clip"
else
    check_fail "clip (import failed)"
fi

if python -c "import ultralytics; print('  ultralytics OK')" 2>/dev/null; then
    check_pass "ultralytics"
else
    check_fail "ultralytics (import failed)"
fi

if python -c "import groundingdino; print('  groundingdino OK')" 2>/dev/null; then
    check_pass "groundingdino"
else
    check_fail "groundingdino (import failed)"
fi

if python -c "import google.generativeai; print('  google.generativeai OK')" 2>/dev/null; then
    check_pass "google.generativeai"
else
    check_fail "google.generativeai (import failed)"
fi

if python -c "import timm; print('  timm OK')" 2>/dev/null; then
    check_pass "timm"
else
    check_fail "timm (import failed)"
fi

# --- 2. GPU check ---
echo ""
echo "[2/5] GPU / CUDA check..."
if python -c "
import torch
assert torch.cuda.is_available(), 'No CUDA'
name = torch.cuda.get_device_name(0)
print(f'  CUDA device: {name}')
" 2>/dev/null; then
    check_pass "CUDA available"
else
    check_fail "CUDA not available"
fi

# --- 3. NAS data path checks ---
echo ""
echo "[3/5] NAS FSC-147 data path checks (${NAS_DATA}/FSC147/)..."
FSC_DIR="${NAS_DATA}/FSC147"
DATA_PATHS=(
    "${FSC_DIR}/images_384_VarV2"
    "${FSC_DIR}/gt_density_map_adaptive_384_VarV2"
    "${FSC_DIR}/Train_Test_Val_FSC_147.json"
    "${FSC_DIR}/annotation_FSC147_384.json"
)
for path in "${DATA_PATHS[@]}"; do
    if [ -e "${path}" ]; then
        check_pass "${path}"
    else
        check_fail "MISSING: ${path}"
    fi
done

# --- 4. Checkpoint files check ---
echo ""
echo "[4/5] Checkpoint files (*.pth) at ${NAS_DATA}/..."
EXPECTED_CKPTS=(
    "${NAS_DATA}/checkpoint_FSC.pth"
    "${NAS_DATA}/checkpoint__finetuning_dino_prompt.pth"
    "${NAS_DATA}/checkpoint__finetuning_yolo.pth"
    "${NAS_DATA}/checkpoint__finetuning_yolo_noprompt.pth"
)
FOUND_COUNT=0
for ckpt in "${EXPECTED_CKPTS[@]}"; do
    if [ -f "${ckpt}" ]; then
        echo "  Found: ${ckpt}"
        FOUND_COUNT=$((FOUND_COUNT + 1))
    else
        echo "  Missing: ${ckpt}"
    fi
done
if [ "${FOUND_COUNT}" -gt 0 ]; then
    check_pass "${FOUND_COUNT}/4 expected checkpoint(s) found"
else
    check_fail "No expected checkpoint files found in ${NAS_DATA}/"
fi

# --- 5. .env / API key check ---
echo ""
echo "[5/5] .env / API key check..."
ENV_FILE="${REPO_DIR}/code/source-code/.env"
if [ -f "${ENV_FILE}" ]; then
    if grep -q "GEMINI_API_KEY" "${ENV_FILE}" 2>/dev/null; then
        check_pass ".env found with GEMINI_API_KEY entry"
    else
        check_fail ".env found but GEMINI_API_KEY not set"
    fi
else
    check_fail ".env missing at ${ENV_FILE} --- copy from .env.example and fill in key"
fi

# --- Summary ---
TOTAL=$((PASS + FAIL))
echo ""
echo "=============================================="
echo " Summary: ${PASS}/${TOTAL} checks passed"
echo "=============================================="
if [ "${FAIL}" -gt 0 ]; then
    echo " ${FAIL} check(s) FAILED --- review output above"
    exit 1
else
    echo " All checks PASSED --- environment ready"
    exit 0
fi
