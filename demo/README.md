# Zero-shot Object Counting — Streamlit Demo

An interactive demo for the VA-Count zero-shot object counting system. Upload any image, type an object class name, and get the predicted count with density map visualization.

Default configuration: **YOLO-World + Rich Prompt** (best speed-accuracy trade-off).

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9+ | 3.12 |
| GPU (CUDA) | Optional (CPU works) | NVIDIA with ≥ 4 GB VRAM |
| Disk space | ~2 GB (models + deps) | ~3 GB (with FSC-147 dataset) |

---

## Setup

### 1. Create a Python virtual environment

```bash
cd demo/
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 2. Install GroundingDINO (editable, must come first)

```bash
cd GroundingDINO
pip install -e .
cd ..
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download GroundingDINO weights (~694 MB)

> Skip this step if the file already exists at `GroundingDINO/weights/groundingdino_swint_ogc.pth`.

```bash
wget -O GroundingDINO/weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### 5. Get a Gemini API key (free)

1. Go to <https://aistudio.google.com/apikey>
2. Click **Create API key** → copy the key.

### 6. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your key:

```
GEMINI_API_KEY=your-gemini-api-key-here
```

> **Do NOT commit** the `.env` file — it is gitignored.

---

## Run the demo

```bash
streamlit run demo_app_advanced.py
```

The browser opens automatically at **http://localhost:8501**.

First launch takes ~2-3 minutes (model loading & caching). Subsequent launches: ~10-30 s.

---

## Usage

1. **Upload an image** (JPG / PNG) via the file uploader.
2. **Enter the object class name** in English (e.g. `oranges`, `dog`, `sheep`).
3. **Adjust settings** in the left sidebar:
   - **Detection Model**: YOLO-World (fast) or GroundingDINO (more accurate)
   - **AI-Enhanced Prompts (Gemini)**: toggle Rich Prompt on/off
   - **Detection Threshold**: lower = more detections (default 0.15)
   - **Positive / Negative Exemplars**: number of example patches
4. Click **Count Objects!** and view:
   - Predicted object count
   - Density heatmap overlay
   - Positive (green) and negative (red) bounding boxes

Images are uploaded through the UI — **no need to place files in any folder**.

---

## File structure

```
demo/
├── README.md                     # This file
├── .env.example                  # Template — copy to .env
├── requirements.txt              # Python dependencies
│
├── demo_app_advanced.py          # Streamlit entry point
├── demo_inference.py             # Model loading & inference
├── demo_pipeline_advanced.py     # Pipeline wrapper
├── demo_visualization.py         # Heatmap / overlay drawing
├── prompt_enhancer.py            # Gemini API for Rich Prompt
├── models_mae_cross.py           # MAE counting model
├── models_crossvit.py            # Cross-attention block
├── util/
│   ├── __init__.py
│   └── pos_embed.py              # Positional embeddings
│
├── data/ → (symlink)             # Checkpoints & dataset
│   ├── checkpoint__finetuning_yolo.pth
│   └── out/classify/best_model.pth
├── GroundingDINO/ → (symlink)    # Detection model + weights
│   └── weights/groundingdino_swint_ogc.pth
└── yolov8x-worldv2.pt → (symlink)
```

> `data/`, `GroundingDINO/`, and `yolov8x-worldv2.pt` are **symlinks** to `code/source-code/` to avoid duplicating large files.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `GEMINI_API_KEY not set` | Missing env variable | Fill in `.env` (see step 6) |
| `No module named GroundingDINO` | Not installed as editable | Run `cd GroundingDINO && pip install -e . && cd ..` |
| `groundingdino_swint_ogc.pth not found` | Weights not downloaded | Run the `wget` command in step 4 |
| First launch very slow (~3 min) | Normal model caching | Subsequent runs use cache |
| Out of memory (GPU) | VRAM insufficient | Reduce exemplar count in sidebar, or use CPU |
| AI enhancement times out | Gemini API unreachable | Disable in sidebar or check internet |
| Broken symlinks | Ran from wrong directory | Always `cd demo/` before running |

---

## Quick verification

```bash
# Check symlinks resolve correctly
ls -lh data/checkpoint__finetuning_yolo.pth
ls -lh GroundingDINO/weights/groundingdino_swint_ogc.pth
ls -lh yolov8x-worldv2.pt

# Test imports
python -c "import demo_inference; print('OK')"

# Run
streamlit run demo_app_advanced.py
```
