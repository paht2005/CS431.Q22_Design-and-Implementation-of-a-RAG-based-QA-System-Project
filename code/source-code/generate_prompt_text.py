"""Bulk Gemini rich-prompt text generator for FSC147 val split.

Produces annotation_FSC147_pos_prompt_text.json, which maps each image in
the requested split to a Gemini-generated visual description:

    {
        "1050.jpg": {"positive_descriptions": ["single bread roll . round . golden brown ."]},
        ...
    }

This file is consumed by owlv2_pos.py / owlv2_neg.py / florence2_pos.py /
florence2_neg.py when the --prompt flag is used (Rich-Prompt mode).

Usage:
    export GEMINI_API_KEY="<your-google-ai-studio-key>"
    python generate_prompt_text.py \\
        --data_root ./data/FSC147 \\
        --split val \\
        --output_file ./data/FSC147/annotation_FSC147_pos_prompt_text.json \\
        --delay 0.5

The script is fully resumable: if --output_file already contains partial
results, already-processed images are skipped.  Checkpoints are written
every 100 images so that a Ctrl+C or quota error loses at most 100 entries.

Security: GEMINI_API_KEY is read from the environment only (os.environ).
The key is never logged or written to disk by this script.
"""

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # python-dotenv is an optional helper
    pass

from prompt_enhancer import enhance_prompt_with_gemini  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk Gemini rich-prompt text generator for FSC147"
    )
    parser.add_argument(
        "--data_root",
        default="./data/FSC147",
        help="Path to the FSC147 root directory (default: ./data/FSC147)",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "train", "test"],
        help="Which split to process (default: val)",
    )
    parser.add_argument(
        "--output_file",
        default="./data/FSC147/annotation_FSC147_pos_prompt_text.json",
        help="Output JSON path (default: ./data/FSC147/annotation_FSC147_pos_prompt_text.json)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to sleep between Gemini API calls for rate limiting (default: 0.5)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_split_images(data_root: str, split: str) -> list[str]:
    """Return sorted list of image filenames for the requested split."""
    split_file = Path(data_root) / "Train_Test_Val_FSC_147.json"
    with open(split_file) as f:
        data = json.load(f)
    return sorted(data[split])


def load_class_map(data_root: str) -> dict[str, str]:
    """Return {img_name: class_name} from ImageClasses_FSC147.txt (tab-separated)."""
    class_file = Path(data_root) / "ImageClasses_FSC147.txt"
    class_map: dict[str, str] = {}
    with open(class_file) as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" in line:
                img_name, class_name = line.split("\t", 1)
            else:
                # Fallback: whitespace-split (first token = img name, rest = class)
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                img_name, class_name = parts
            class_map[img_name.strip()] = class_name.strip()
    return class_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Security: validate API key exists in env before doing any work.
    # prompt_enhancer.py will raise on import if the key is missing, but we
    # add an explicit early check here for a clearer error message.
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Export it in your shell before running:\n"
            "  export GEMINI_API_KEY=<your-key>\n"
            "Never commit the key to the repository."
        )

    data_root = args.data_root
    output_file = Path(args.output_file)

    # Load split image list and class map
    target_images = load_split_images(data_root, args.split)
    class_map = load_class_map(data_root)

    # Resume support: load partial results if output file already exists
    result_map: dict[str, dict] = {}
    if output_file.exists():
        with open(output_file) as f:
            result_map = json.load(f)
        print(f"Resuming: {len(result_map)} images already done.")

    images_dir = Path(data_root) / "images_384_VarV2"
    total = len(target_images)
    checkpoint_interval = 100

    for i, img_name in enumerate(target_images, start=1):
        if img_name in result_map:
            continue  # already processed

        img_path = images_dir / img_name
        class_name = class_map.get(img_name, "object")

        image = Image.open(img_path).convert("RGB")
        enhanced = enhance_prompt_with_gemini(image, class_name)

        result_map[img_name] = {"positive_descriptions": [enhanced]}
        print(f"[{i}/{total}] {img_name} ({class_name}) → {enhanced[:60]}")

        time.sleep(args.delay)

        # Checkpoint write every 100 images
        if i % checkpoint_interval == 0:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(result_map, f, indent=2)
            print(f"  [checkpoint] {len(result_map)} entries saved.")

    # Final save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result_map, f, indent=2)

    print(f"Done. {len(result_map)} entries saved to {output_file}")


if __name__ == "__main__":
    main()
