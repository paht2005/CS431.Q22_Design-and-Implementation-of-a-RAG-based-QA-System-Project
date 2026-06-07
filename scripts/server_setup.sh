#!/usr/bin/env bash
# Bootstrap the project environment on counting@192.168.6.200 (lab server, mmlab2024).
# Run from the repo root after SSH-ing into the server.
# Usage: bash scripts/server_setup.sh
#
# Prerequisites on server:
#   - Conda (Miniconda / Anaconda) available
#   - NAS mounted at /mnt/mmlab2024nas
#   - Git available

set -euo pipefail

REPO_URL="https://github.com/paht2005/CS338.Q21_Zero-shot-Object-Coutning-with-Good-Examplers.git"
REPO_DIR="${HOME}/cs338-counting"
CONDA_ENV="cs338"
NAS_DATA="/mnt/mmlab2024nas/counting"

echo "=============================================="
echo " VA-Count Server Setup --- counting@192.168.6.200"
echo "=============================================="

# --- 1. Check NAS mount ---
echo ""
echo "[1/8] Checking NAS mount at ${NAS_DATA}..."
if [ ! -d "${NAS_DATA}" ]; then
    echo "  ERROR: NAS directory ${NAS_DATA} does not exist."
    echo "  Ensure /mnt/mmlab2024nas is mounted before running this script."
    exit 1
fi
echo "  NAS OK: ${NAS_DATA}"

# --- 2. Clone or update repo ---
echo ""
echo "[2/8] Cloning / updating repository..."
if [ -d "${REPO_DIR}/.git" ]; then
    echo "  Repo exists --- pulling latest..."
    cd "${REPO_DIR}"
    git pull
else
    echo "  Cloning to ${REPO_DIR}..."
    git clone "${REPO_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}"
fi

# --- 3. Create Conda environment ---
echo ""
echo "[3/8] Creating Conda env '${CONDA_ENV}' (Python 3.10)..."
# shellcheck source=/dev/null
CONDA_BASE="${CONDA_BASE:-/opt/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
    echo "  Conda env '${CONDA_ENV}' already exists --- skipping creation"
else
    conda create -n "${CONDA_ENV}" python=3.10 -y
fi
conda activate "${CONDA_ENV}"
echo "  Python: $(python --version)"

# --- 4. Install PyTorch with CUDA (conda-managed, cuda 12.1 default) ---
echo ""
echo "[4/8] Installing PyTorch (pytorch-cuda=12.1 --- change if server CUDA differs)..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# --- 5. Install GroundingDINO (editable) ---
echo ""
echo "[5/8] Installing GroundingDINO (editable build)..."
cd "${REPO_DIR}/code/source-code/GroundingDINO"
pip install -e .
cd "${REPO_DIR}"

# --- 6. Install main project dependencies ---
echo ""
echo "[6/8] Installing project dependencies from requirements.txt..."
pip install -r code/source-code/requirements.txt

# --- 7. Install CLIP ---
echo ""
echo "[7/8] Installing CLIP..."
pip install git+https://github.com/openai/CLIP.git


# --- 7b. Set HuggingFace cache to NAS (prevents /home disk fill) ---
echo ""
echo "[7b] Configuring HuggingFace model cache -> NAS..."
grep -q 'HF_HOME' ~/.bashrc || echo 'export HF_HOME=/mnt/mmlab2024nas/counting/hf_cache' >> ~/.bashrc
grep -q 'TRANSFORMERS_CACHE' ~/.bashrc || echo 'export TRANSFORMERS_CACHE=/mnt/mmlab2024nas/counting/hf_cache' >> ~/.bashrc
export HF_HOME=/mnt/mmlab2024nas/counting/hf_cache
export TRANSFORMERS_CACHE=/mnt/mmlab2024nas/counting/hf_cache
mkdir -p "${HF_HOME}"
echo "  HF_HOME -> ${HF_HOME}"

# --- 7c. Install OWL-v2 / Florence-2 dependencies ---
echo ""
echo "[7c] Installing transformers, accelerate, einops, timm..."
pip install transformers accelerate einops timm
echo "  OWL-v2 / Florence-2 deps installed."
# --- 8. Check NAS data paths ---
echo ""
echo "[8/8] Checking NAS data paths at ${NAS_DATA}/FSC147/..."
DATA_PATHS=(
    "${NAS_DATA}/FSC147/images_384_VarV2"
    "${NAS_DATA}/FSC147/gt_density_map_adaptive_384_VarV2"
    "${NAS_DATA}/FSC147/Train_Test_Val_FSC_147.json"
    "${NAS_DATA}/FSC147/annotation_FSC147_384.json"
)
ALL_OK=true
for path in "${DATA_PATHS[@]}"; do
    if [ -e "${path}" ]; then
        echo "  OK: ${path}"
    else
        echo "  MISSING: ${path}"
        ALL_OK=false
    fi
done
if [ "${ALL_OK}" = false ]; then
    echo ""
    echo "  WARNING: FSC-147 data incomplete on NAS."
    echo "  Upload data to ${NAS_DATA}/FSC147/ with structure:"
    echo "    FSC147/"
    echo "    ├── images_384_VarV2/          (~6135 images)"
    echo "    ├── gt_density_map_adaptive_384_VarV2/"
    echo "    ├── Train_Test_Val_FSC_147.json"
    echo "    └── annotation_FSC147_384.json"
fi

# --- Checkpoint check ---
echo ""
echo "Checking checkpoints at ${NAS_DATA}/..."
EXPECTED_CKPTS=(
    "checkpoint_FSC.pth"
    "checkpoint__finetuning_dino_prompt.pth"
    "checkpoint__finetuning_yolo.pth"
    "checkpoint__finetuning_yolo_noprompt.pth"
)
for ckpt in "${EXPECTED_CKPTS[@]}"; do
    if [ -f "${NAS_DATA}/${ckpt}" ]; then
        echo "  Found: ${ckpt}"
    else
        echo "  Missing: ${ckpt} --- upload to ${NAS_DATA}/"
    fi
done

# --- .env reminder ---
echo ""
ENV_FILE="${REPO_DIR}/code/source-code/.env"
if [ -f "${ENV_FILE}" ]; then
    echo ".env found at ${ENV_FILE}"
else
    echo "WARNING: No .env found. Copy .env.example -> .env and set GEMINI_API_KEY."
fi

echo ""
echo "=============================================="
echo " Setup complete!"
echo " Activate env : conda activate ${CONDA_ENV}"
echo " Data (NAS)   : ${NAS_DATA}"
echo "=============================================="
