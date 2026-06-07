#!/usr/bin/env bash
# Generate exemplars across all supported detector pipelines.
#
# Usage:
#   bash scripts/generate_exemplars.sh [MODE]
#
# Modes:
#   dino            GroundingDINO positive + negative (Phase 1 baseline)
#   yolo            YOLO-World positive (with/without prompt) + negative (Phase 1)
#   owlv2           OWL-v2 positive + negative, raw class-name query (DET-03)
#   florence2       Florence-2 positive + negative, raw class-name query (DET-03)
#   owlv2-prompt    OWL-v2 positive + negative, Rich Prompts via --prompt (DET-04)
#   florence2-prompt Florence-2 positive + negative, Rich Prompts via --prompt (DET-04)
#   all-new         owlv2 + florence2 + owlv2-prompt + florence2-prompt (Phase 3 all)
#   all             dino + yolo + all-new (everything)
#
# Phase 3 — 8 generation commands (DET-03 + DET-04):
#
#   Raw (DET-03):
#     bash scripts/generate_exemplars.sh owlv2
#     bash scripts/generate_exemplars.sh florence2
#
#   Rich-Prompt (DET-04) — requires annotation_FSC147_pos_prompt_text.json first:
#     bash scripts/generate_exemplars.sh owlv2-prompt
#     bash scripts/generate_exemplars.sh florence2-prompt
#
# Common options passed through to each detector script:
#   TEXT_FILE   path to ImageClasses_FSC147.txt  (default: ./data/FSC147/ImageClasses_FSC147.txt)
#   DATASET     path to images_384_VarV2/         (default: ./data/FSC147/images_384_VarV2/)
#   DATA_ROOT   path to FSC147 root               (default: ./data/FSC147)
#   SPLIT       val | train | test                 (default: val)
#
# Override example:
#   SPLIT=test bash scripts/generate_exemplars.sh owlv2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/code/source-code"
MODE="${1:-all}"

TEXT_FILE="${TEXT_FILE:-./data/FSC147/ImageClasses_FSC147.txt}"
DATASET="${DATASET:-./data/FSC147/images_384_VarV2/}"
DATA_ROOT="${DATA_ROOT:-./data/FSC147}"
SPLIT="${SPLIT:-val}"

PROMPT_TEXT_FILE="${DATA_ROOT}/annotation_FSC147_pos_prompt_text.json"

cd "${SRC}"

# ---------------------------------------------------------------------------
# Phase 1 - GroundingDINO
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "dino" || "${MODE}" == "all" ]]; then
    echo "=== Generating POSITIVE exemplars with GroundingDINO ==="
    python grounding_pos.py

    echo "=== Generating NEGATIVE exemplars with GroundingDINO ==="
    python grounding_neg.py
fi

# ---------------------------------------------------------------------------
# Phase 1 - YOLO-World
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "yolo" || "${MODE}" == "all" ]]; then
    echo "=== Generating POSITIVE exemplars with YOLO-World (with prompt) ==="
    python yolo_pos_withPrompt.py

    echo "=== Generating POSITIVE exemplars with YOLO-World (without prompt) ==="
    python yolo_pos_withoutPrompt.py

    echo "=== Generating NEGATIVE exemplars with YOLO-World ==="
    python yolo_neg.py
fi

# ---------------------------------------------------------------------------
# Phase 3 - OWL-v2 raw (DET-03)
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "owlv2" || "${MODE}" == "all-new" || "${MODE}" == "all" ]]; then
    echo "=== [DET-03] Generating POSITIVE exemplars with OWL-v2 (raw) ==="
    python owlv2_pos.py \
        --text_file "${TEXT_FILE}" \
        --dataset_path "${DATASET}" \
        --output_file "${DATA_ROOT}/annotation_FSC147_pos_owlv2.json" \
        --split "${SPLIT}"

    echo "=== [DET-03] Generating NEGATIVE exemplars with OWL-v2 (raw) ==="
    python owlv2_neg.py \
        --text_file "${TEXT_FILE}" \
        --dataset_path "${DATASET}" \
        --output_file "${DATA_ROOT}/annotation_FSC147_neg_owlv2.json" \
        --split "${SPLIT}"
fi

# ---------------------------------------------------------------------------
# Phase 3 - Florence-2 raw (DET-03)
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "florence2" || "${MODE}" == "all-new" || "${MODE}" == "all" ]]; then
    echo "=== [DET-03] Generating POSITIVE exemplars with Florence-2 (raw) ==="
    python florence2_pos.py \
        --text_file "${TEXT_FILE}" \
        --dataset_path "${DATASET}" \
        --output_file "${DATA_ROOT}/annotation_FSC147_pos_florence2.json" \
        --split "${SPLIT}"

    echo "=== [DET-03] Generating NEGATIVE exemplars with Florence-2 (raw) ==="
    python florence2_neg.py \
        --text_file "${TEXT_FILE}" \
        --dataset_path "${DATASET}" \
        --output_file "${DATA_ROOT}/annotation_FSC147_neg_florence2.json" \
        --split "${SPLIT}"
fi

# ---------------------------------------------------------------------------
# Phase 3 - OWL-v2 Rich Prompts (DET-04)
# Prerequisite: annotation_FSC147_pos_prompt_text.json must exist (Plan 03-03)
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "owlv2-prompt" || "${MODE}" == "all-new" || "${MODE}" == "all" ]]; then
    if [[ ! -f "${PROMPT_TEXT_FILE}" ]]; then
        echo "ERROR: Rich-Prompt text file not found: ${PROMPT_TEXT_FILE}"
        echo "Run generate_prompt_text.py first (Plan 03-03)."
        exit 1
    fi

    echo "=== [DET-04] Generating POSITIVE exemplars with OWL-v2 (Rich Prompt) ==="
    python owlv2_pos.py \
        --text_file "${TEXT_FILE}" \
        --dataset_path "${DATASET}" \
        --output_file "${DATA_ROOT}/annotation_FSC147_pos_owlv2_prompt.json" \
        --prompt \
        --split "${SPLIT}"

    echo "=== [DET-04] Generating NEGATIVE exemplars with OWL-v2 (Rich Prompt) ==="
    python owlv2_neg.py \
        --text_file "${TEXT_FILE}" \
        --dataset_path "${DATASET}" \
        --output_file "${DATA_ROOT}/annotation_FSC147_neg_owlv2_prompt.json" \
        --prompt \
        --split "${SPLIT}"
fi

# ---------------------------------------------------------------------------
# Phase 3 - Florence-2 Rich Prompts (DET-04)
# Prerequisite: annotation_FSC147_pos_prompt_text.json must exist (Plan 03-03)
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "florence2-prompt" || "${MODE}" == "all-new" || "${MODE}" == "all" ]]; then
    if [[ ! -f "${PROMPT_TEXT_FILE}" ]]; then
        echo "ERROR: Rich-Prompt text file not found: ${PROMPT_TEXT_FILE}"
        echo "Run generate_prompt_text.py first (Plan 03-03)."
        exit 1
    fi

    echo "=== [DET-04] Generating POSITIVE exemplars with Florence-2 (Rich Prompt) ==="
    python florence2_pos.py \
        --text_file "${TEXT_FILE}" \
        --dataset_path "${DATASET}" \
        --output_file "${DATA_ROOT}/annotation_FSC147_pos_florence2_prompt.json" \
        --prompt \
        --split "${SPLIT}"

    echo "=== [DET-04] Generating NEGATIVE exemplars with Florence-2 (Rich Prompt) ==="
    python florence2_neg.py \
        --text_file "${TEXT_FILE}" \
        --dataset_path "${DATASET}" \
        --output_file "${DATA_ROOT}/annotation_FSC147_neg_florence2_prompt.json" \
        --prompt \
        --split "${SPLIT}"
fi

echo ""
echo "=== Exemplar generation complete (mode: ${MODE}) ==="
