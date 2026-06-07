"""
Visualization utilities for VA-Count Demo
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Color schemes
COLOR_RAW = (255, 165, 0)  # Orange for raw detections
COLOR_ACCEPTED = (0, 191, 255)  # Deep sky blue for accepted
COLOR_REJECTED = (128, 128, 128)  # Gray for rejected
COLOR_POSITIVE = (0, 255, 0)  # Green for positive exemplars
COLOR_NEGATIVE = (255, 0, 0)  # Red for negative exemplars

BOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_boxes_on_image(
    image, boxes, scores=None, color=COLOR_RAW, labels=None, thickness=BOX_THICKNESS
):
    """
    Draw bounding boxes on image

    Args:
        image: PIL Image or numpy array
        boxes: numpy array of shape [N, 4] (x1, y1, x2, y2)
        scores: optional list of scores
        color: BGR color tuple
        labels: optional list of labels
        thickness: box thickness

    Returns:
        numpy array (BGR format for display)
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_np = image.copy()

    if len(boxes) == 0:
        return img_np

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # Draw rectangle
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        label = ""
        if labels is not None and i < len(labels):
            label = labels[i]
        elif scores is not None and i < len(scores):
            label = f"{scores[i]:.2f}"

        if label:
            label_size, _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
            y_label = y1 - 10 if y1 - 10 > 20 else y1 + 20

            # Draw background for text
            cv2.rectangle(
                img_np,
                (x1, y_label - label_size[1] - 4),
                (x1 + label_size[0] + 4, y_label + 4),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                img_np,
                label,
                (x1 + 2, y_label),
                FONT,
                FONT_SCALE,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return img_np


def visualize_detection_stage(image, boxes, scores, stage_name="Detection"):
    """Visualize raw detection stage"""
    labels = [f"#{i + 1}: {s:.2f}" for i, s in enumerate(scores)]
    img_with_boxes = draw_boxes_on_image(image, boxes, color=COLOR_RAW, labels=labels)
    return cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)


def visualize_classifier_stage(
    image, accepted_boxes, rejected_boxes, classifier_scores
):
    """Visualize binary classifier results"""
    if isinstance(image, Image.Image):
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_np = image.copy()

    # Draw rejected boxes (gray)
    if len(rejected_boxes) > 0:
        idx_start = len(accepted_boxes)
        rejected_labels = [
            f"✗ {classifier_scores[idx_start + i]:.2f}"
            for i in range(len(rejected_boxes))
        ]
        img_np = draw_boxes_on_image(
            img_np, rejected_boxes, color=COLOR_REJECTED, labels=rejected_labels
        )

    # Draw accepted boxes (blue)
    if len(accepted_boxes) > 0:
        accepted_labels = [
            f"✓ {classifier_scores[i]:.2f}" for i in range(len(accepted_boxes))
        ]
        img_np = draw_boxes_on_image(
            img_np, accepted_boxes, color=COLOR_ACCEPTED, labels=accepted_labels
        )

    return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)


def visualize_exemplars(image, pos_boxes, pos_scores, neg_boxes, neg_scores):
    """Visualize selected exemplars"""
    if isinstance(image, Image.Image):
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_np = image.copy()

    # Draw negative exemplars (red)
    if len(neg_boxes) > 0:
        neg_labels = [f"Neg-{i + 1}: {s:.2f}" for i, s in enumerate(neg_scores)]
        img_np = draw_boxes_on_image(
            img_np, neg_boxes, color=COLOR_NEGATIVE, labels=neg_labels, thickness=3
        )

    # Draw positive exemplars (green)
    if len(pos_boxes) > 0:
        pos_labels = [f"Pos-{i + 1}: {s:.2f}" for i, s in enumerate(pos_scores)]
        img_np = draw_boxes_on_image(
            img_np, pos_boxes, color=COLOR_POSITIVE, labels=pos_labels, thickness=3
        )

    return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)


def extract_crops_for_display(image, boxes, max_crops=3):
    """Extract crop images for display"""
    crops = []

    for i, box in enumerate(boxes[:max_crops]):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)

        if isinstance(image, Image.Image):
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)
            crop = image.crop((x1, y1, x2, y2))
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x2 = min(img_rgb.shape[1], x2)
            y2 = min(img_rgb.shape[0], y2)
            crop = Image.fromarray(img_rgb[y1:y2, x1:x2])

        crops.append(crop)

    return crops


def create_density_heatmap(density_map, colormap="jet"):
    """
    Create colored heatmap from density map

    Args:
        density_map: numpy array (H, W)
        colormap: matplotlib colormap name

    Returns:
        RGB image (H, W, 3) in range [0, 255]
    """
    if density_map is None:
        return None

    # Normalize density map
    density_normalized = density_map.copy()
    vmax = density_normalized.max()
    if vmax > 0:
        density_normalized = density_normalized / vmax

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(density_normalized)

    # Convert to RGB [0, 255]
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    return rgb


def create_overlay_image(original_image, density_map, alpha=0.5, colormap="jet"):
    """
    Create overlay of density map on original image

    Args:
        original_image: PIL Image or numpy array
        density_map: numpy array (384, 384)
        alpha: overlay transparency (0=original, 1=heatmap only)
        colormap: matplotlib colormap

    Returns:
        RGB numpy array
    """
    if density_map is None:
        if isinstance(original_image, Image.Image):
            return np.array(original_image)
        return original_image

    # Convert original to numpy
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image)
    else:
        img_np = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Resize to match density map
    img_resized = cv2.resize(img_np, (384, 384))

    # Create heatmap
    heatmap_rgb = create_density_heatmap(density_map, colormap)

    # Blend
    overlay = cv2.addWeighted(img_resized, 1 - alpha, heatmap_rgb, alpha, 0)

    return overlay


def visualize_gt_density(gt_density_map, colormap="jet"):
    """Visualize ground truth density map"""
    return create_density_heatmap(gt_density_map, colormap)


def create_comparison_view(original_image, pred_density, gt_density, alpha=0.5):
    """
    Create side-by-side comparison of predicted vs GT

    Returns:
        RGB image with pred on left, GT on right
    """
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image)
    else:
        img_np = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img_np, (384, 384))

    # Create overlays
    pred_overlay = create_overlay_image(img_resized, pred_density, alpha)
    gt_overlay = create_overlay_image(img_resized, gt_density, alpha)

    # Concatenate horizontally
    comparison = np.hstack([pred_overlay, gt_overlay])

    return comparison


def create_stage_summary_image(image, results):
    """
    Create a summary visualization showing all stages

    Args:
        image: original PIL Image
        results: dict from full_counting_pipeline

    Returns:
        Large summary image showing all stages
    """
    stages = []

    # Stage 1: Raw detections
    if len(results.get("raw_boxes", [])) > 0:
        stage1 = visualize_detection_stage(
            image, results["raw_boxes"], results["raw_scores"]
        )
        stages.append(("Raw Detections", stage1))

    # Stage 2: After classifier
    if (
        len(results.get("accepted_boxes", [])) > 0
        or len(results.get("rejected_boxes", [])) > 0
    ):
        stage2 = visualize_classifier_stage(
            image,
            results["accepted_boxes"],
            results["rejected_boxes"],
            results["classifier_scores"],
        )
        stages.append(("After Classifier", stage2))

    # Stage 3: Selected exemplars
    if len(results.get("pos_boxes", [])) > 0:
        stage3 = visualize_exemplars(
            image,
            results["pos_boxes"],
            results["pos_scores"],
            results.get("neg_boxes", []),
            results.get("neg_scores", []),
        )
        stages.append(("Selected Exemplars", stage3))

    return stages


def add_text_annotation(
    image,
    text,
    position="top",
    bg_color=(0, 0, 0),
    text_color=(255, 255, 255),
    font_scale=0.8,
):
    """Add text annotation to image"""
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()

    if len(img_np.shape) == 2:  # Grayscale
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    h, w = img_np.shape[:2]

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, 2)

    # Determine position
    if position == "top":
        y_pos = 30
    elif position == "bottom":
        y_pos = h - 20
    else:
        y_pos = h // 2

    x_pos = (w - text_w) // 2

    # Draw background rectangle
    cv2.rectangle(
        img_np,
        (x_pos - 10, y_pos - text_h - 10),
        (x_pos + text_w + 10, y_pos + baseline + 10),
        bg_color,
        -1,
    )

    # Draw text
    cv2.putText(
        img_np, text, (x_pos, y_pos), FONT, font_scale, text_color, 2, cv2.LINE_AA
    )

    return img_np
