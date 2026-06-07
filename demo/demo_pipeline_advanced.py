"""
Advanced pipeline wrapper for VA-Count with expansion prompt and adjustable binary classifier threshold.
"""

import numpy as np
from demo_inference import full_counting_pipeline, expand_prompts


def advanced_counting_pipeline(
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
    detection_threshold=0.05,
    top_k_pos=3,
    top_k_neg=3,
    prompt_for_detection=None,
    binary_classifier_threshold=0.8,
    neg_prompt=None,
):
    """
    Wrapper for full_counting_pipeline with support for:
    - prompt_for_detection: can be a string or list (expansion prompt)
    - binary_classifier_threshold: override threshold for single/multiple object
    """
    # Handle expansion prompt
    if prompt_for_detection is not None:
        # If list, use first element for display, join for YOLO, or pass as is for GroundingDINO
        if isinstance(prompt_for_detection, list):
            prompt = prompt_for_detection[0]
        else:
            prompt = prompt_for_detection
    else:
        prompt = text_prompt

    # Patch SINGLE_OBJECT_THRESHOLD if needed
    import demo_inference

    old_threshold = demo_inference.SINGLE_OBJECT_THRESHOLD
    demo_inference.SINGLE_OBJECT_THRESHOLD = binary_classifier_threshold
    try:
        # Run full_counting_pipeline up to negative exemplar selection
        results = full_counting_pipeline(
            image=image,
            text_prompt=prompt,
            counting_model=counting_model,
            detection_model_type=detection_model_type,
            grounding_dino_model=grounding_dino_model,
            yolo_model=yolo_model,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            binary_classifier=binary_classifier,
            device=device,
            detection_threshold=detection_threshold,
            top_k_pos=top_k_pos,
            top_k_neg=top_k_neg,
            neg_prompt=neg_prompt,
        )
      
   
    finally:
        demo_inference.SINGLE_OBJECT_THRESHOLD = old_threshold
    return results
