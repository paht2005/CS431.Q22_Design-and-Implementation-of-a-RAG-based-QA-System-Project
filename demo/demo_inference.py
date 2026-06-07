import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from pathlib import Path
import pathlib
import clip
from ultralytics import YOLOWorld
import time
import inflect
import platform

# Fix pathlib for Windows
if platform.system() == "Windows":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# Import project modules
import models_mae_cross

# Constants
MAX_HW = 384
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]
DENSITY_SCALE = 60
SINGLE_OBJECT_THRESHOLD = 0.8
ALPHA = 0.2  # GroundingDINO weight
BETA = 0.8  # CLIP weight

# Initialize inflect engine for singularization
p = inflect.engine()


# ============================================
# BINARY CLASSIFIER
# ============================================


class ClipClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim=512):
        super(ClipClassifier, self).__init__()
        self.clip_model = clip_model
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(clip_model.visual.output_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images).float()
        x = self.fc(image_features)
        x = torch.relu(x)
        logits = self.classifier(x)
        return logits


# ============================================
# MODEL LOADING
# ============================================


def load_counting_model(
    checkpoint_path, device="cuda", model_name="mae_vit_base_patch16"
):
    """Load VA-Count model"""
    print(f"Loading counting model from {checkpoint_path}...")
    model = models_mae_cross.__dict__[model_name](norm_pix_loss=False)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print("✅ Counting model loaded")
    return model


def load_yolo_model(model_path, device="cuda"):
    """Load YOLOv8-World model"""
    print(f"Loading YOLOWorld from {model_path}...")
    model = YOLOWorld(model_path)
    # Move model to the specified device
    if device == "cuda" and torch.cuda.is_available():
        model.to(device)
    print(f"✅ YOLOWorld loaded on {device}")
    return model


def load_grounding_dino(device="cuda"):
    """Load GroundingDINO model"""
    print("Loading GroundingDINO...")
    from groundingdino.util.inference import load_model

    config_file = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "./GroundingDINO/weights/groundingdino_swint_ogc.pth"

    model = load_model(config_file, checkpoint_path, device=device)
    print("✅ GroundingDINO loaded")
    return model


def load_clip_model(device="cuda"):
    """Load CLIP model"""
    print("Loading CLIP...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    print("✅ CLIP loaded")
    return clip_model, preprocess


def load_binary_classifier(clip_model, device="cuda"):
    """Load binary classifier for single/multiple object detection"""
    print("Loading Binary Classifier...")
    classifier = ClipClassifier(clip_model).to(device)

    weights_path = "./data/out/classify/best_model.pth"
    if Path(weights_path).exists():
        classifier.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        print("✅ Binary Classifier loaded")
    else:
        print("⚠️ Binary Classifier weights not found, using untrained model")

    classifier.eval()
    return classifier


# ============================================
# DETECTION STAGE
# === Utility functions for YOLO detection ===
from PIL import Image


def resize_image_keep_aspect(image, max_size=1024):
    """
    Resize image keeping aspect ratio
    Args:
        image: PIL Image
        max_size: Maximum dimension (width or height)
    Returns:
        resized_image: PIL Image
        scale_factor: float
        imgsz: int (recommended YOLO imgsz, multiple of 32)
    """
    w, h = image.size
    if w <= max_size and h <= max_size:
        max_dim = max(w, h)
        imgsz = ((max_dim + 31) // 32) * 32
        return image, 1.0, min(imgsz, max_size)
    scale = min(max_size / w, max_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    max_dim = max(new_w, new_h)
    imgsz = ((max_dim + 31) // 32) * 32
    imgsz = min(imgsz, max_size)
    return resized, scale, imgsz


def expand_prompts(base_prompt, enable_expansion=True):
    """Expand single prompt to multiple variations"""
    if not enable_expansion:
        return [base_prompt]
    prompts = [base_prompt]
    if not base_prompt.startswith("a "):
        prompts.append(f"a {base_prompt}")
    prompts.append(f"a photo of a {base_prompt}")
    prompts.append(f"multiple {base_prompt}")
    return prompts


# ============================================


def detect_with_grounding_dino(
    image, text_prompt, model, device="cuda", box_threshold=0.05, text_threshold=0.05
):
    """
    Detect objects using GroundingDINO with filtering for dense objects
    Returns: boxes, scores, detection_time
    """
    from groundingdino.util.inference import load_image, predict
    from torchvision.ops import box_convert
    import tempfile
    import os

    start_time = time.time()

    # Singularization
    singular_name = p.singular_noun(text_prompt)
    if not singular_name:
        singular_name = text_prompt

    # Build prompt
    # caption = f"multiple single {singular_name} ."
    caption = f" {singular_name} ."
    print("Using caption for GroundingDINO:", caption)
    # Save PIL image to temporary file for GroundingDINO
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Load image using GroundingDINO's function (returns tensor)
        image_source, image_tensor = load_image(tmp_path)
        pil_img = Image.open(tmp_path).convert("RGB")
        h, w, _ = image_source.shape

        # Run detection
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    detection_time = time.time() - start_time

    # Convert boxes from cxcywh to xyxy
    H, W = image.height, image.width

    # Convert to tensor if needed
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    # Filter out large group boxes (following original logic)
    filtered_boxes = []
    filtered_logits = []

    for i, box in enumerate(boxes):
        # Convert from normalized cxcywh to pixel coordinates for filtering
        cx, cy, w, h = box.tolist()

        # # Skip large boxes that likely represent groups (>30% of image)
        # if w > 0.3 or h > 0.3:
        #     continue

        # Convert to pixel coordinates
        box_pixel = box * torch.Tensor([W, H, W, H])
        box_xyxy = box_convert(
            boxes=box_pixel.unsqueeze(0), in_fmt="cxcywh", out_fmt="xyxy"
        )

        filtered_boxes.append(box_xyxy.squeeze().cpu().numpy())
        filtered_logits.append(
            logits[i].cpu().item() if torch.is_tensor(logits[i]) else logits[i]
        )

    return np.array(filtered_boxes), np.array(filtered_logits), detection_time


def detect_with_yolo(image, text_prompt, model, device="cuda", conf_threshold=0.01):
    """
    Detect objects using YOLOv8-World
    Returns: boxes, scores, detection_time
    """

    start_time = time.time()

    # Resize image and get scale, imgsz
    resized_image, scale_factor, imgsz = resize_image_keep_aspect(image, max_size=1024)

    # Expand prompt
    prompts = expand_prompts(text_prompt, enable_expansion=True)

    # Set YOLOWorld classes
    try:
        if hasattr(model, "to"):
            model.to(device)
        model.set_classes(prompts)
    except Exception as e:
        print(f"YOLO set_classes error: {e}")
        model.set_classes([text_prompt])

    # Convert PIL to numpy
    image_np = np.array(resized_image)

    # Run detection
    results = model.predict(image_np, conf=conf_threshold, imgsz=imgsz, verbose=False)

    detection_time = time.time() - start_time

    boxes = []
    scores = []
    labels = []

    if len(results) > 0 and results[0].boxes is not None:
        boxes_data = results[0].boxes.xyxy.cpu().numpy()
        scores_data = results[0].boxes.conf.cpu().numpy()
        labels_data = (
            results[0].boxes.cls.cpu().numpy()
            if hasattr(results[0].boxes, "cls")
            else np.zeros_like(scores_data)
        )

        W, H = resized_image.size

        for box, score, label in zip(boxes_data, scores_data, labels_data):
            x1, y1, x2, y2 = box
            w_box, h_box = x2 - x1, y2 - y1
            # Skip if too large (>50% of image) or too small (<5 pixels)
            if w_box > W / 2 or h_box > H / 2 or w_box < 5 or h_box < 5:
                continue
            # Scale box back to original image size
            if scale_factor != 1.0:
                x1 /= scale_factor
                y1 /= scale_factor
                x2 /= scale_factor
                y2 /= scale_factor
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(int(label))

    return np.array(boxes), np.array(scores), np.array(labels), detection_time


# ============================================
# BINARY CLASSIFIER STAGE
# ============================================


def filter_single_objects(image, boxes, classifier, clip_preprocess, device="cuda"):
    """
    Filter boxes to keep only single objects
    Returns: accepted_boxes, rejected_boxes, all_scores
    """
    if len(boxes) == 0:
        return np.array([]), np.array([]), []

    accepted = []
    rejected = []
    all_scores = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Crop and classify
        crop = image.crop((x1, y1, x2, y2))
        crop_tensor = clip_preprocess(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = classifier(crop_tensor)
            prob = torch.softmax(logits, dim=1)[0, 1].item()  # P(single object)

        all_scores.append(prob)

        if prob > SINGLE_OBJECT_THRESHOLD:
            accepted.append(box)
        else:
            rejected.append(box)

    return np.array(accepted), np.array(rejected), all_scores


# ============================================
# EXEMPLAR SELECTION
# ============================================


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x, y, w, h] format"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    intersection_x1 = max(x1, x2)
    intersection_y1 = max(y1, y2)
    intersection_x2 = min(x1 + w1, x2 + w2)
    intersection_y2 = min(y1 + h1, y2 + h2)
    intersection_area = max(intersection_x2 - intersection_x1, 0) * max(
        intersection_y2 - intersection_y1, 0
    )
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def select_exemplars(
    image,
    boxes,
    detection_scores,
    text_prompt,
    clip_model,
    clip_preprocess,
    device="cuda",
    top_k_pos=3,
):
    """
    Select positive exemplars using CLIP (matching grounding_pos.py)
    Returns: pos_boxes, pos_scores
    """
    if len(boxes) == 0:
        return np.array([]), []

    # Singularize and prepare text prompt
    singular_name = p.singular_noun(text_prompt)
    if not singular_name:
        singular_name = text_prompt

    # Encode text using singular form (matching grounding_pos.py)
    text_input = f"a photo of a single {singular_name}"
    text_tokens = clip.tokenize([text_input]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Score each box
    scored_boxes = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        crop_tensor = clip_preprocess(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(crop_tensor).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        clip_sim = (image_features @ text_features.T).item()

        # Combined score (detection + CLIP) matching grounding_pos.py
        det_score = detection_scores[i] if i < len(detection_scores) else 0.0
        combined_score = (ALPHA * det_score) + (BETA * clip_sim)

        scored_boxes.append((box, combined_score, clip_sim))

    if len(scored_boxes) == 0:
        return np.array([]), []

    # Sort by score
    scored_boxes.sort(key=lambda x: x[1], reverse=True)

    # Select top-K positive
    pos_boxes = [box for box, _, _ in scored_boxes[:top_k_pos]]
    pos_scores = [score for _, score, _ in scored_boxes[:top_k_pos]]

    return np.array(pos_boxes), pos_scores




# ============================================
# COUNTING INFERENCE
# ============================================


def preprocess_image(image):
    """Preprocess image to 384x384"""
    W, H = image.size
    new_H = min(H, MAX_HW)
    new_W = min(W, MAX_HW)

    scale_factor_H = new_H / H
    scale_factor_W = new_W / W

    resized_image = image.resize((new_W, new_H), Image.LANCZOS)

    # Pad to 384x384
    padded = Image.new("RGB", (MAX_HW, MAX_HW), (0, 0, 0))
    padded.paste(resized_image, (0, 0))

    return padded, scale_factor_H, scale_factor_W


def extract_exemplar_crops(image, boxes, crop_size=64):
    """Extract and preprocess exemplar crops following the notebook's logic:

    1. Crop the box from the 384x384 image.
    2. Resize to 64x64 (the final size expected by the model).

    Model `forward_decoder` expects boxes with shape ``[N, shot_num, 3, 64, 64]``.
    """
    if len(boxes) == 0:
        return torch.zeros(1, 3, crop_size, crop_size)

    # Convert to tensor for processing
    to_tensor = transforms.ToTensor()
    
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)

        if x2 > x1 and y2 > y1:
            # Step 1: Crop from the source image
            crop = image.crop((x1, y1, x2, y2))
            
            # Step 2: Resize to 64x64 (final size expected by the model)
            crop_64 = crop.resize((64, 64), Image.LANCZOS)
            
            # Convert to tensor (NO normalization)
            crop_tensor = to_tensor(crop_64)
            crops.append(crop_tensor)

    if len(crops) == 0:
        return torch.zeros(1, 3, crop_size, crop_size)

    return torch.stack(crops)


def run_counting_inference(
    model,
    image,
    pos_boxes,
    device="cuda",
    shot_num=3,
):
    """Run VA-Count model inference following the notebook's logic.

    Args:
        model: VA-Count model.
        image: Input PIL Image.
        pos_boxes: Positive exemplar boxes (N, 4) ``[x1, y1, x2, y2]``.
        device: ``cuda`` or ``cpu``.
        shot_num: Number of exemplars (typically 3).

    Returns:
        density_map_np: Final density map (numpy array).
        count: Final estimated count.
        processed_image: Preprocessed image.
    """
    # Preprocess image - resize to 384x384
    processed_image = image.resize((384, 384), Image.LANCZOS)

    # Convert to tensor WITHOUT normalization (matching the notebook)
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(processed_image).unsqueeze(0).to(device)

    # ========================================
    # POSITIVE EXEMPLARS - Theo notebook logic
    # ========================================
    pos_exemplar_crops = extract_exemplar_crops(
        processed_image, pos_boxes, crop_size=64
    )
    pos_exemplar_crops = pos_exemplar_crops.to(device)

    # Pad or truncate to match shot_num
    if pos_exemplar_crops.shape[0] < shot_num:
        padding = pos_exemplar_crops[-1:].repeat(
            shot_num - pos_exemplar_crops.shape[0], 1, 1, 1
        )
        pos_exemplar_crops = torch.cat([pos_exemplar_crops, padding], dim=0)
    elif pos_exemplar_crops.shape[0] > shot_num:
        pos_exemplar_crops = pos_exemplar_crops[:shot_num]

    # Add batch dimension: [1, shot_num, 3, 64, 64] - model expects 64x64
    pos_exemplar_crops = pos_exemplar_crops.unsqueeze(0)

    # Run model with positive exemplars
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            density_map = model(image_tensor, pos_exemplar_crops, shot_num)

    # ========================================
    # FINAL COUNT - divide by 60 (matching the notebook)
    # ========================================
    count = torch.abs(density_map.sum()).item() / 60.0
    count = max(0, count)

    # Convert to numpy
    density_map_np = density_map.squeeze().cpu().numpy()

    return (
        density_map_np,
        count,
        processed_image,
    )


# ============================================
# FULL PIPELINE
# ============================================


def full_counting_pipeline(
    image,
    text_prompt,
    counting_model,
    detection_model_type,
    grounding_dino_model,
    yolo_model,
    clip_model,
    clip_preprocess,
    binary_classifier,
    device="cuda",
    # Adjustable parameters
    detection_threshold=0.05,
    top_k_pos=3,
    top_k_neg=3,
    neg_prompt=None,
):
    """
    Full counting pipeline with detailed intermediate results

    Returns dict with:
        - raw_boxes, raw_scores, detection_time
        - accepted_boxes, rejected_boxes, classifier_scores
        - pos_boxes, pos_scores
        - neg_boxes, neg_scores
        - density_map, count, processed_image
    """
    results = {}

    # Stage 1: Detection
    print(f"🔍 Detection with {detection_model_type}...")
    if detection_model_type == "GroundingDINO":
        raw_boxes, raw_scores, det_time = detect_with_grounding_dino(
            image,
            text_prompt,
            grounding_dino_model,
            device,
            box_threshold=detection_threshold,
            text_threshold=detection_threshold,
        )
    else:  # YOLOv8-World
        raw_boxes, raw_scores, _, det_time = detect_with_yolo(
            image, text_prompt, yolo_model, device, conf_threshold=detection_threshold
        )

    results["raw_boxes"] = raw_boxes
    results["raw_scores"] = raw_scores
    results["detection_time"] = det_time

    if len(raw_boxes) == 0:
        print("⚠️ No objects detected → Count set to 0")
        results["accepted_boxes"] = np.array([])
        results["rejected_boxes"] = np.array([])
        results["classifier_scores"] = []
        results["pos_boxes"] = np.array([])
        results["pos_scores"] = []
        results["density_map"] = None
        results["count"] = 0
        results["processed_image"] = None
        return results

    # Stage 2: Binary Classifier
    print(f"✅ Binary Classifier filtering...")
    accepted_boxes, rejected_boxes, classifier_scores = filter_single_objects(
        image, raw_boxes, binary_classifier, clip_preprocess, device
    )

    results["accepted_boxes"] = accepted_boxes
    results["rejected_boxes"] = rejected_boxes
    results["classifier_scores"] = classifier_scores

    if len(accepted_boxes) == 0:
        print("⚠️ No single objects found (all filtered by classifier) → Count set to 0")
        results["pos_boxes"] = np.array([])
        results["pos_scores"] = []
        results["density_map"] = None
        results["count"] = 0
        results["processed_image"] = None
        return results

    # Stage 3: Positive Exemplar Selection
    print(f"⭐ Selecting positive exemplars...")
    pos_boxes, pos_scores = select_exemplars(
        image,
        accepted_boxes,
        raw_scores[: len(accepted_boxes)],
        text_prompt,
        clip_model,
        clip_preprocess,
        device,
        top_k_pos=top_k_pos,
    )

    results["pos_boxes"] = pos_boxes
    results["pos_scores"] = pos_scores

    if len(pos_boxes) == 0:
        print("⚠️ No positive exemplars selected → Count set to 0")
        results["neg_boxes"] = np.array([])
        results["neg_scores"] = []
        results["density_map"] = None
        results["count"] = 0
        results["processed_image"] = None
        return results


    # Stage 4: Counting
    print(f"🔢 Running counting inference...")

    density_map, count, processed_image = run_counting_inference(
        counting_model,
        image,
        pos_boxes,
        device,
        shot_num=top_k_pos,
    )

    results["density_map"] = density_map
    results["count"] = count
    results["processed_image"] = processed_image

    print(f"✅ Pipeline complete! Count: {count:.2f}")

    return results
