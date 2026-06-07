<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology (UIT)" width="400">
  </a>
</p>

<h1 align="center"><b>Zero-shot Object Counting with Good Exemplars</b></h1>
<h3 align="center">Enhanced with Rich Prompts and YOLO-World</h3>
<h4 align="center">CS431.Q22 – Deep Learning and Applications</h4>
<h4 align="center">Instructor: PhD. Nguyen Vinh Tiep</h4>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/GroundingDINO-000000?style=for-the-badge" />
  <img src="https://img.shields.io/badge/YOLO--World-00A67E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Gemini_Prompts-4285F4?style=for-the-badge" />
</p>

---


# **Project Overview**

> This repository contains the implementation of a **Zero-shot Object Counting system**  
> based on **VA-Count (Zero-shot Object Counting with Good Exemplars)**, extended with:
>
> - **Rich Prompts**: Large Language Model–generated prompts (Gemini) + CLIP-style re-ranking  
>   to obtain higher-quality positive and negative exemplars.
> - **YOLO-World**: A fast open-vocabulary detector used to replace GroundingDINO in the
>   exemplar extraction stage, dramatically reducing latency while preserving accuracy.
>
> The system targets **generalized object counting in unseen classes** using **FSC147**,  
> and is evaluated with standard metrics (**MAE, RMSE**) and detailed latency measurements.

The full written report is stored at: **[`docs/report/Report.pdf`](docs/report/Report.pdf)**.  
Heavy artifacts (full checkpoints, raw `wandb` runs, full FSC147 dataset, GroundingDINO /
YOLO-World weights) are hosted **outside Git** to keep the repository under GitHub's
100 MB file limit. The shared folder below contains the large files that are intentionally
ignored by `.gitignore` and therefore not stored in this repository:

> **Full project artifacts (OneDrive):**
> https://nklod-my.sharepoint.com/personal/phatxinhchao_nklod_onmicrosoft_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fphatxinhchao%5Fnklod%5Fonmicrosoft%5Fcom%2FDocuments%2FUIT%2Fzero%2Dshot%2Dobject%2Dcounting%2Dwith%2Dgood%2Dexamplers&ga=1
>
> This folder includes the full FSC147 dataset, checkpoints, raw `wandb` runs, and detector
> weights referenced in [`docs/report/Report.pdf`](docs/report/Report.pdf) and omitted from Git.

---

<p align="center">
  <img src="images/thumbnail.png" alt="Project thumbnail" width="400">
</p>

---

## Group 07 - Team Information

**Instructor:** PhD. Nguyen Vinh Tiep

| No. | Student ID | Full Name        | Role   | Github                                                   | Email                   |
|----:|:----------:|------------------|--------|----------------------------------------------------------|-------------------------|
| 1   | 23521143   | Nguyen Cong Phat | Leader | [paht2005](https://github.com/paht2005) (primary account), [phatcnguyenKMS](https://github.com/phatcnguyenKMS) (secondary account)| 23521143@gm.uit.edu.vn |
| 2   | 23521168   | Nguyen Le Phong  | Member | [kllp031](https://github.com/kllp031)                     | 23521168@gm.uit.edu.vn |
| 3   | 23520213   | Vu Viet Cuong    | Member | [Kun05-AI](https://github.com/Kun05-AI)                  | 23520213@gm.uit.edu.vn |


> **About this project.** This repository is the team's own re-implementation
> of the ECCV 2024 **VA-Count** paper, extended with two independent additions
> from the literature: **Rich Prompts** (Zhu et al., 2025) for higher-quality
> exemplar generation, and **YOLO-World** (Cheng et al., 2024) as a
> drop-in replacement for GroundingDINO in the exemplar-extraction stage. The
> codebase is structured around four reproducible artifacts: training, testing,
> exemplar generation, and an interactive Streamlit demo. Every MAE / RMSE /
> latency number quoted in this README is traceable to a `wandb` run archived
> under `experiments/exp{2,3,4,5}/wandb/` — see
> [`docs/RESULTS.md`](docs/RESULTS.md) for the exact provenance per number.

---

## **Table of Contents**

- [Repository Structure](#repository-structure)
- [Problem Statement](#problem-statement)
- [Method Overview](#method-overview)
  - [Baseline VA-Count (EEM + NSM)](#baseline-va-count-eem--nsm)
  - [Rich Prompt Extension](#rich-prompt-extension)
  - [YOLO-World Extension](#yolo-world-extension)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Experiments & Metrics](#experiments--metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Demo Application](#demo-application)
- [Experimental Results (High-level)](#experimental-results-high-level)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)
- [Further Reading](#further-reading)

---

## **Repository Structure**

```text
cs338-zero-shot-object-counting-with-good-examplers/
├── README.md                 # Main project documentation (this file)
├── Makefile                  # Top-level build/run shortcuts (make help)
├── LICENSE                   # MIT License
├── requirements.txt          # Root-level Python dependencies
├── env.template              # Environment variable template (.env)
│
├── demo/                     # demo section
|
├── code/                     # Main implementation
│   ├── README.md
│   └── source-code/          # VA-Count + Rich Prompt + YOLO-World
│       ├── README.md
│       ├── data/              # FSC147 dataset (images, density maps, annotations)
│       ├── GroundingDINO/     # GroundingDINO code + weights
│       ├── util/              # Dataset loader, misc utilities
│       ├── models_*.py        # MAE / CrossViT model definitions
│       ├── FSC_pretrain.py    # Pretraining script
│       ├── FSC_train.py       # Training / fine-tuning script
│       ├── FSC_test.py        # Evaluation script (MAE, RMSE)
│       ├── prompt_enhancer.py # Gemini-based Rich Prompt generation
│       ├── grounding_*.py     # GroundingDINO-based exemplar extraction
│       ├── yolo_*.py          # YOLO-World–based exemplar extraction
│       ├── demo_*.py          # Streamlit demo & visualization scripts
│       └── requirements.txt   # Python dependencies
│
├── configs/                   # YAML run configurations for training & evaluation
│   ├── README.md
│   ├── train_baseline.yaml
│   ├── train_finetune_dino_prompt.yaml
│   ├── train_finetune_yolo.yaml
│   └── test_baseline.yaml
│
├── scripts/                   # Shell helpers for reproducibility
│   ├── README.md
│   ├── setup_env.sh           # Conda env creation + dependency install
│   ├── download_data.sh       # FSC147 dataset & checkpoint download
│   ├── generate_exemplars.sh  # Exemplar generation (GroundingDINO / YOLO)
│   └── run_evaluation.sh      # Full evaluation suite
│
├── experiments/               # Archived experiment runs and wandb logs
│   ├── README.md
│   ├── exp2/                  # VA-Count baseline runs
│   ├── exp3/                  # YOLO-World ablations
│   ├── exp4/                  # Mixed VA-Count / Rich Prompt / YOLO runs
│   └── exp5/                  # Additional Rich Prompt / YOLO ablations
│
├── docs/                      # Reports, slides, and references
│   ├── README.md
│   ├── CONTRIBUTIONS.md       # Per-member contribution split
│   ├── RESULTS.md             # Numerical results & provenance
│   ├── report/                # LaTeX report → Report.pdf
│   ├── slide/           # LaTeX presentation slides
│   └── references/            # Reference materials
│
├── images/                    # Images for GitHub (thumbnails, screenshots)
│   └── thumbnail.png
│
└── draft/                     # Early development drafts (historical)
    └── ...                    # Original upstream code before restructuring
```

- **`code/`** – Everything needed to **run the models**:
  baseline VA-Count, Rich Prompt pipeline, YOLO-World exemplar extraction,
  demo applications and visualization scripts.
  See [`code/README.md`](code/README.md) and [`code/source-code/README.md`](code/source-code/README.md).

- **`configs/`** – YAML configuration files consumed by training and evaluation scripts.
  See [`configs/README.md`](configs/README.md).

- **`scripts/`** – Shell scripts wrapping common workflows (setup, download, exemplar
  generation, evaluation). See [`scripts/README.md`](scripts/README.md).

- **`experiments/`** – Archived `wandb` runs, visualizations, and data snapshots supporting
  the numbers in `Report.pdf`. See [`experiments/README.md`](experiments/README.md).

- **`docs/`** – Course reports, presentation slides, contribution records, and references.
  See [`docs/README.md`](docs/README.md).

- **`images/`** – Assets for the GitHub repository (thumbnails, screenshots).

- **`draft/`** – Early development drafts and original upstream code before project
  restructuring. Kept for historical reference.

---

## **Problem Statement**

Counting objects in natural images is challenging due to:

- Wide variability in **object scale, density, and occlusion**.
- The need to generalize to **unseen classes** without additional annotations.
- The cost of **dense labeling** (dot annotations, bounding boxes) at scale.

The project addresses **Zero-shot Object Counting**:

> Given an RGB image and a **text prompt** describing the target class  
> (e.g. `"person"`, `"oranges"`), estimate:
> - A **density map** over the image.
> - The **total count** of objects that match the prompt.

Under the constraints:

- Only a **single target class** is counted per run.
- Prompts are **English**, concrete, countable nouns.
- Objects are visually observable in real-world scenes.

---

## **Method Overview**

The implementation follows VA-Count and extends it with two major components.

### Baseline VA-Count (EEM + NSM)

Baseline pipeline (Section 4.1 in the report):

- **Exemplar Enhancement Module (EEM)**:
  - Uses **GroundingDINO** to propose **positive** and **negative** exemplar boxes.
  - Filters boxes via:
    - **Deduplication** across positive/negative streams (IoU-based).
    - **Binary classifier** for **single-object filtering**.
  - Selects top exemplars for both positive and negative branches.

- **Noise Suppression Module (NSM)**:
  - MAE-based image encoder + interaction module + decoder.
  - Produces:
    - Positive density map `Dp`
    - Negative density map `Dn`
  - Trained with:
    - **Contrastive loss** `LC(Dp, Dg, Dn)`
    - **Density MSE loss** `LD(Dp, Dg)`
    - `L_total = LC + LD`

### Rich Prompt Extension

To reduce ambiguity and improve exemplar quality, the project introduces **Rich Prompts**:

- Uses **Gemini** to generate:
  - **Positive prompts**: visual definition of a *single* target instance
    (shape, color, material, etc.).
  - **Negative prompts**: background and distractor descriptions that must *not*
    describe the target class.
- Uses these prompts as **queries** for GroundingDINO.
- Adds **CLIP ViT-B/32 re-ranking** over the proposed boxes:
  - Patches from candidate boxes vs. class name text embeddings.
  - Selects top-k (e.g. 5) patches with highest CLIP similarity as final exemplars.

### YOLO-World Extension

GroundingDINO is accurate but **slow and heavy**.  
To make exemplar extraction real-time friendly, the project explores **YOLO-World**:

- Replaces GroundingDINO proposals with **YOLO-World detections**.
- Leverages YOLO-World’s **Prompt-then-Detect** design and **RepPAN** backbone.
- Achieves significantly higher FPS while preserving acceptable accuracy.
- Combined with Rich Prompts, the model reaches a good trade-off between:
  - **Accuracy** (MAE/RMSE on FSC147)
  - **Latency** (seconds per image in the demo)

---

## **Key Features**

- Zero-shot object counting on **unseen classes** via text prompts.
- Strong baseline **VA-Count** implementation (EEM + NSM).
- **Rich Prompt** generation and CLIP-based re-ranking for robust exemplars.
- **YOLO-World** integration for fast exemplar extraction (20× speedup region).
- Full experimental pipeline:
  - FSC147 dataset preparation
  - Training, pretraining, and testing scripts
  - Demo and visualization tools
  - Archived experiment logs (`wandb`, visualizations)

---

## **Dataset**

The project uses **FSC147**, a standard dataset for generalized object counting:

- ~6,135 images, **147 classes**.
- High intra-class variation in:
  - Object size
  - Density
  - Background complexity
- Provides:
  - Class labels
  - Dot annotations
  - Train/Val/Test splits

Official dataset (UIT-hosted Kaggle copy) used in this project:

- Kaggle dataset: `https://www.kaggle.com/datasets/xuncngng/fsc147-0`

Expected structure under `code/source-code/data/FSC147/` is documented in  
`code/source-code/README.md` (images, density maps, annotations, splits).

### Dataset in this repository

Because FSC147 is relatively large and subject to its own license, **this repository does not contain the full dataset**.  
Instead, only a **very small sample** is tracked to illustrate the expected directory layout:

- `code/source-code/data/FSC147/sample/`
  - Contains **2–5 example images** from FSC147.
  - Each sample image is paired with its corresponding **dot-annotation / metadata file**
    following the **official FSC147 folder and file naming conventions**.

To run training or full evaluation you must:

1. Download the full dataset from the Kaggle link above.
2. Unzip it and place the content under:

   - `code/source-code/data/FSC147/`

3. Make sure the folder hierarchy (images, annotations, density maps, splits, etc.) matches
   what is described in `code/source-code/README.md`.

The `.gitignore` is configured so that:

- The **full FSC147 dataset** under `code/source-code/data/FSC147/` is **ignored by git**.
- Only the tiny **`sample/`** subfolder (2–5 images + annotations) is included in version control
  as a reference example.

---

## **What is tracked in Git vs. kept locally**

To make the repository lightweight and friendly for GitHub (no files larger than 100 MB),
we separate **source code and small samples** from **heavy artifacts**.

- **Tracked in Git (safe to clone / view on GitHub)**:
  - All Python source code under `code/source-code/` (models, training, testing, demos, utilities).
  - Configuration files and scripts.
  - Project documentation (`README.md`, `docs/`, `images/`).
  - A tiny FSC147 sample under `code/source-code/data/FSC147/sample/`
    (2–5 images + their annotations) to illustrate the expected data layout.

- **Ignored by Git (large or environment-specific artifacts)**:
  - Full FSC147 dataset under `code/source-code/data/FSC147/` (only `sample/` is kept).
  - All checkpoints and large model weights (`*.pth`, `*.pt`, `*.ckpt`, `*.onnx`).
  - Training / evaluation outputs:
    - `code/source-code/output/`, `code/source-code/output_transfer/`,
      `code/source-code/data/out/`, `demo_outputs/`.
    - Experiment outputs under `experiments/**/data/out/` and `experiments/**/visualizations/`.
  - Visualization artifacts:
    - `code/source-code/visualizations/`, `code/source-code/visualizations_test/`.
  - Experiment tracking logs:
    - `wandb/`, `code/source-code/wandb/`, `experiments/**/wandb/`.
  - Heavy experiment-only weights:
    - `experiments/**/GroundingDINO/weights/`, `experiments/**/yolo*.pt`.
  - Very large visualization notebooks under `experiments/**/notebook/`.
  - Any temporary copies of FSC147 placed under experiments/**/data/FSC147/ are also ignored by git and are meant only for local experiments.

All of these ignore rules are defined in `.gitignore` so that **pushing to GitHub never uploads
large datasets or checkpoints**, and contributors only interact with the **code, docs, and small
sample data**.

---

## **Experiments & Metrics**

The project follows the report:

- **Metrics**:
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Squared Error)
- **Comparisons**:
  - Baseline VA-Count
  - VA-Count + Rich Prompts
  - VA-Count + YOLO-World
  - VA-Count + YOLO-World + Rich Prompts
- **Artifacts**:
  - Numerical results and plots: logged via `wandb` in `experiments/exp*/wandb/`.
  - Qualitative results and failure cases: stored in `experiments/exp*/visualizations/`.

---

## **Installation**

1. **Clone the repository**

```bash
git clone https://github.com/paht2005/CS338.Q21_Zero-shot-Object-Coutning-with-Good-Examplers.git
cd CS338.Q21_Zero-shot-Object-Coutning-with-Good-Examplers
```

2. **Create and activate a virtual environment (recommended)**

```bash
python -m venv env
source env/bin/activate          # Linux/macOS
# env\Scripts\activate           # Windows
```

> The `env/` directory is already listed in `.gitignore` and will not be
> committed to the repository.

3. **Configure environment variables**

```bash
cp env.template .env
# Edit .env and fill in your API keys (GEMINI_API_KEY, etc.)
```

> The `.env` file is gitignored and must **never** be committed.
> See [`env.template`](env.template) for all available variables.

4. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

> The root `requirements.txt` targets modern PyTorch (`torch>=2.0`) and
> works on macOS (CPU / MPS), Linux CPU, and CUDA >= 11.7. To reproduce the
> exact MAE / RMSE / latency numbers reported in `docs/report/Report.pdf`
> on a CUDA 11.6 box, install the pinned CUDA 11.6 variant instead:
>
> ```bash
> pip install -r code/source-code/requirements-cuda116.txt
> ```

5. **Install GroundingDINO (editable)**

```bash
cd code/source-code/GroundingDINO
pip install -e .
cd ../../..
```

6. **Prepare the FSC147 dataset**

Follow the **Dataset Preparation** section in `code/source-code/README.md`:

- Download FSC147 from the official repository.
- Place it under `code/source-code/data/FSC147/`.
- Run the split-generation script to create `train.txt`, `val.txt`, `test.txt`.

7. **Download pretrained checkpoints**

Also described in `code/source-code/README.md`:

- Main VA-Count checkpoint.
- Binary classifier checkpoint (optional).
- MAE pretrained backbone.
- YOLO-World weights (if not already present).

---

## **Usage**

All commands below are run from:

```bash
cd code/source-code
```

### 1. Baseline VA-Count training / evaluation

- **Train / fine-tune**:

```bash
python FSC_train.py --data_path ./data/FSC147 \
    --anno_file annotation_FSC147_pos.json \
    --output_dir ./data/out/finetune_pos
```

- **Evaluate** (compute MAE/RMSE):

```bash
python FSC_test.py \
    --data_path ./data/FSC147 \
    --split test \
    --resume ./data/checkpoint_FSC.pth
```

### 2. Generate exemplars with GroundingDINO

```bash
# Positive exemplars
python grounding_pos.py --root_path ./data/FSC147/

# Negative exemplars
python grounding_neg.py --root_path ./data/FSC147/
```

### 3. Generate exemplars with YOLO-World

```bash
# With prompts
python yolo_pos_withPrompt.py --root_path ./data/FSC147/

# Without prompts
python yolo_pos_withoutPrompt.py --root_path ./data/FSC147/

# Negative examples
python yolo_neg.py --root_path ./data/FSC147/
```

### 4. Use Rich Prompts (prompt_enhancer)

Configure your Gemini API key (see `prompt_enhancer.py`), then run the prompt enhancement:

```bash
python prompt_enhancer.py --data_path ./data/FSC147
```

This will generate enhanced prompts that can be consumed by the GroundingDINO / YOLO-World
exemplar-generation scripts.

---

## **Demo Application**

The interactive demo is a **Streamlit** app at
`code/source-code/demo_app_advanced.py`. Configure your Gemini API key first
(copy `env.template` to `.env` and fill in `GEMINI_API_KEY`), then run:

```bash
cd code/source-code
streamlit run demo_app_advanced.py
```

The app lets you:

- Upload an image and type an English class name (e.g. `oranges`).
- Pick the extractor (**YOLO-World** or **GroundingDINO**) and toggle
  **Rich Prompt** on/off.
- See the predicted count, the density-map overlay, the chosen positive /
  negative bounding boxes, and a comparison with ground-truth when available.

Supporting modules consumed by the Streamlit app:

- `demo_inference.py` — model loading, prompt expansion, single-image
  inference helpers.
- `demo_pipeline_advanced.py` — end-to-end VA-Count + Rich Prompt + YOLO
  pipeline.
- `demo_visualization.py` — overlay / heat-map / box drawing utilities.

For headless / CLI evaluation of the network use `FSC_test.py` instead — see
[`code/source-code/README.md`](code/source-code/README.md) for full details.

---

## **Experimental Results (High-level)**

On the **FSC-147 test split**:

| Model                                        | MAE ↓     | RMSE ↓    | Demo time (s/img) ↓ |
|----------------------------------------------|-----------|-----------|---------------------|
| VA-Count (baseline)                          | 17.99     | 129.39    | 1.4710              |
| VA-Count + Rich Prompt                       | **17.80** | 129.69    | 5.7578              |
| VA-Count + YOLO-World                        | 19.03     | 131.55    | **0.6006**          |
| **VA-Count + YOLO-World + Rich Prompt**      | 17.91     | 130.98    | 2.4054              |

- **Rich Prompt** improves accuracy slightly on the GroundingDINO baseline
  (MAE 17.99 → 17.80) and substantially on the YOLO-World variant
  (MAE 19.03 → 17.91).
- **YOLO-World alone** is 2.5× faster at demo-time but loses ~1 MAE versus
  GroundingDINO; pairing it with Rich Prompt recovers the accuracy.
- **`YOLO-World + Rich Prompt`** is the **default deployed configuration** —
  near-baseline MAE (17.91) at acceptable interactive latency (~2.4 s/image).

Full per-table breakdown (counting accuracy, exemplar-extraction wall-clock,
demo latency, failure-case taxonomy, dataset / checkpoint provenance) is in
[`docs/RESULTS.md`](docs/RESULTS.md) and Chapter 3 of
[`docs/report/Report.pdf`](docs/report/Report.pdf).

---

## References

## References

- [1] H. Zhu, S. Li, J. Yuan, Z. Yang, Y. Guo, W. Liu, X. Zhong, and S. He,  
  “Expanding zero-shot object counting with rich prompts,”  
  arXiv preprint arXiv:2505.15398, 2025.  
  [Online]. Available: https://arxiv.org/abs/2505.15398
- [2] H. Zhu, J. Yuan, Z. Yang, Y. Guo, Z. Wang, X. Zhong, and S. He,  
  “Zero-shot object counting with good exemplars,”  
  in *European Conference on Computer Vision (ECCV)*, 2024.  
  [Online]. Available: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00812.pdf
- [3] C. Liu, Y. Zhong, A. Zisserman, and W. Xie,  
  “CounTR: Transformer-based generalised visual counting,”  
  in *British Machine Vision Conference (BMVC)*, 2022.  
  [Online]. Available: https://pdf-files.bmvc2022.org/0370.pdf
- [4] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan,  
  “YOLO-World: Real-time open-vocabulary object detection,”  
  in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024.  
  [Online]. Available: https://openaccess.thecvf.com/content/CVPR2024/papers/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.pdf
  “Yolo-world: Real-time open-vocabulary object detection,” 2024.  
  [Online]. Available: https://arxiv.org/abs/2401.17270

---

## **Limitations & Future Work**

- In very dense scenes or with heavily overlapping objects, the density map can
  under-count objects.
- For fragmented objects (e.g. windows with many panes, eyeglasses), the model may
  over-count sub-parts as separate instances.
- The MAE + density map paradigm can still produce plausible counts even when
  exemplars fail, indicating some reliance on dataset bias.

Future directions:

- Improve handling of **dense and fragmented objects**.
- Explore alternative counting paradigms beyond density maps (e.g. relational counting).
- Further tighten the coupling between exemplars and prediction to reduce bias.

---

## **License**
This project is developed for **academic purposes** under the course  
**CS338.Q21 – Pattern Recognition** at the **University of Information Technology (UIT)**.

Released under the **MIT License**.
See the [LICENSE.txt](./LICENSE.txt) file for details.

---

## **Further Reading**

- Final report (Vietnamese): [`docs/report/Report.pdf`](docs/report/Report.pdf)
  — 26-page bilingual write-up with full methodology, tables and failure
  analysis. Build instructions: [`docs/report/README.md`](docs/report/README.md).
- Numerical results & provenance: [`docs/RESULTS.md`](docs/RESULTS.md).
- Per-member contributions: [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md).
- Source-code level instructions: [`code/source-code/README.md`](code/source-code/README.md).
- Figure provenance for the LaTeX report:
  [`docs/report/figures/README.md`](docs/report/figures/README.md).
- Archived experiment runs (`wandb`, qualitative outputs):
  [`experiments/README.md`](experiments/README.md) and the per-experiment
  READMEs under `experiments/exp{2,3,4,5}/`.
