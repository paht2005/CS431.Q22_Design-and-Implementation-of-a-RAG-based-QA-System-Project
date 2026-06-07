"""
Generate POSITIVE exemplar annotations using OWL-v2 (Google).
Mirrors grounding_pos.py but replaces GroundingDINO with OWL-v2.

Usage:
    # Val split, no Rich Prompt
    python owlv2_pos.py \
        --text_file ./data/FSC147/ImageClasses_FSC147.txt \
        --dataset_path ./data/FSC147/images_384_VarV2/ \
        --output_file ./data/FSC147/annotation_FSC147_pos_owlv2.json \
        --split val

    # Val split, with Rich Prompt
    python owlv2_pos.py \
        --text_file ./data/FSC147/ImageClasses_FSC147.txt \
        --dataset_path ./data/FSC147/images_384_VarV2/ \
        --output_file ./data/FSC147/annotation_FSC147_pos_owlv2_prompt.json \
        --prompt \
        --split val

Requirements:
    pip install transformers accelerate
    (CLIP and ultralytics already in requirements.txt)
"""

import torch
import os
import json
import argparse
import clip
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import inflect

device = "cuda" if torch.cuda.is_available() else "cpu"
SCORE_THRESHOLD = 0.05  # mirrors BOX_THRESHOLD in grounding_pos.py
BINARY_THRESHOLD = 0.8
TOP_K_DEFAULT = 3

p = inflect.engine()


# ---------------------------------------------------------------------------
# Binary classifier (identical to grounding_pos.py)
# ---------------------------------------------------------------------------
class ClipClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim=512):
        super().__init__()
        self.clip_model = clip_model.to(device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(clip_model.visual.output_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, images):
        with torch.no_grad():
            feats = self.clip_model.encode_image(images).float().to(device)
        return self.classifier(F.relu(self.fc(feats)))


def is_valid_patch(patch, classifier, preprocess, threshold=BINARY_THRESHOLD):
    if patch.size[0] <= 0 or patch.size[1] <= 0:
        return False
    t = preprocess(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(classifier(t), dim=1)
    return probs[0, 1].item() > threshold


# ---------------------------------------------------------------------------
# OWL-v2 detection
# ---------------------------------------------------------------------------
def detect_owlv2(image_pil, text_queries, processor, model, threshold=SCORE_THRESHOLD):
    """
    Args:
        image_pil: PIL Image
        text_queries: list of strings, e.g. ["apple", "a photo of an apple"]
    Returns:
        list of [x, y, w, h] boxes in pixel coords (float)
    """
    texts = [text_queries]  # OWL-v2 expects [[q1, q2, ...]]
    inputs = processor(text=texts, images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_pil.size[::-1]])  # (H, W)
    results = processor.image_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    W, H = image_pil.size
    boxes_xywh = []
    for box in results["boxes"].tolist():
        x1, y1, x2, y2 = box
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(W), x2), min(float(H), y2)
        if x2 > x1 and y2 > y1:
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
    return boxes_xywh


# ---------------------------------------------------------------------------
# CLIP re-ranking (identical logic to grounding_pos.py)
# ---------------------------------------------------------------------------
def clip_score(patch_pil, class_name, clip_model_l14, preprocess_l14):
    text = clip.tokenize([class_name]).to(device)
    img_t = preprocess_l14(patch_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        img_f = clip_model_l14.encode_image(img_t).float()
        txt_f = clip_model_l14.encode_text(text).float()
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    return (img_f @ txt_f.T).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate positive exemplar annotations with OWL-v2"
    )
    parser.add_argument("--text_file", required=True,
                        help="ImageClasses_FSC147.txt (image_name TAB class_name)")
    parser.add_argument("--dataset_path", required=True,
                        help="Directory containing FSC-147 images")
    parser.add_argument("--output_file", required=True,
                        help="Output JSON annotation file")
    parser.add_argument("--prompt", action="store_true",
                        help="Use Rich Prompt positive descriptions")
    parser.add_argument("--prompt_file",
                        default="./data/FSC147/annotation_FSC147_pos_prompt_text.json",
                        help="JSON with {image_name: {positive_descriptions: [...]}}")
    parser.add_argument("--top_k", type=int, default=TOP_K_DEFAULT,
                        help="Max exemplars to keep per image")
    parser.add_argument("--split_file",
                        default="./data/FSC147/Train_Test_Val_FSC_147.json")
    parser.add_argument("--split", default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--score_threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--model_id",
                        default="google/owlv2-base-patch16-ensemble",
                        help="HuggingFace model ID for OWL-v2")
    args = parser.parse_args()

    # Load data split
    with open(args.split_file) as f:
        splits = json.load(f)
    target_images = set(splits[args.split])

    # Load class names
    class_map = {}
    with open(args.text_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                class_map[parts[0]] = parts[1]

    # Load Rich Prompt texts if requested
    prompt_map = {}
    if args.prompt:
        with open(args.prompt_file) as f:
            prompt_map = json.load(f)
        print(f"Loaded {len(prompt_map)} Rich Prompt entries.")

    # Load OWL-v2
    print(f"Loading OWL-v2 ({args.model_id}) ...")
    processor = Owlv2Processor.from_pretrained(args.model_id)
    owlv2_model = Owlv2ForObjectDetection.from_pretrained(args.model_id).to(device)
    owlv2_model.eval()
    print("OWL-v2 loaded.")

    # Load CLIP models
    print("Loading CLIP models ...")
    clip_b32, preprocess_b32 = clip.load("ViT-B/32", device)
    clip_l14, preprocess_l14 = clip.load("ViT-L/14", device)
    clip_b32.eval()
    clip_l14.eval()

    # Load binary classifier
    print("Loading binary classifier ...")
    classifier = ClipClassifier(clip_b32).to(device)
    weights_path = "./data/out/classify/best_model.pth"
    if os.path.exists(weights_path):
        classifier.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=False)
        )
        print("Binary classifier weights loaded.")
    else:
        print("WARNING: Binary classifier weights not found, running without filtering.")
    classifier.eval()

    # Collect images for this split
    image_files = sorted([
        f for f in os.listdir(args.dataset_path)
        if f in target_images and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Processing {len(image_files)} images (split='{args.split}') ...")

    annotations = {}
    for i, img_name in enumerate(image_files, 1):
        img_path = os.path.join(args.dataset_path, img_name)
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        class_name = class_map.get(img_name, "object")
        singular = p.singular_noun(class_name) or class_name

        # Build text queries
        if args.prompt and img_name in prompt_map:
            queries = prompt_map[img_name].get("positive_descriptions", [])
            if not queries:
                queries = [singular]
        else:
            # Three query formulations to improve recall (same strategy as grounding_pos.py)
            queries = [singular, f"a photo of {singular}", f"a single {singular}"]

        boxes = detect_owlv2(
            image, queries, processor, owlv2_model, threshold=args.score_threshold
        )

        # Binary-classifier filter + CLIP re-ranking
        scored = []
        for box in boxes:
            x, y, w_, h_ = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            crop = image.crop((x, y, x + w_, y + h_))
            if is_valid_patch(crop, classifier, preprocess_b32):
                s = clip_score(crop, singular, clip_l14, preprocess_l14)
                scored.append((s, box))

        scored.sort(key=lambda t: t[0], reverse=True)
        valid_boxes = [b for _, b in scored[: args.top_k]]

        annotations[img_name] = {"H": H, "W": W, "boxes": valid_boxes}
        print(f"  [{i}/{len(image_files)}] {img_name}: "
              f"{len(boxes)} detected -> {len(valid_boxes)} kept")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"\nSaved {len(annotations)} entries to {args.output_file}")


if __name__ == "__main__":
    main()
