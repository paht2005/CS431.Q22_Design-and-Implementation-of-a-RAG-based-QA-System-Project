"""
VA-Count Advanced Demo Application
Zero-Shot Object Counting with Good Exemplars (Advanced)
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import time

# Import custom modules
from demo_inference import (
    load_counting_model,
    load_yolo_model,
    load_grounding_dino,
    load_clip_model,
    load_binary_classifier,
    expand_prompts,
)
from demo_pipeline_advanced import advanced_counting_pipeline
from prompt_enhancer import enhance_prompt_with_gemini, enhance_prompt_simple
from demo_visualization import (
    visualize_detection_stage,
    visualize_classifier_stage,
    visualize_exemplars,
    extract_crops_for_display,
    create_density_heatmap,
    create_overlay_image,
    visualize_gt_density,
    create_comparison_view,
    add_text_annotation,
)

# Page config
st.set_page_config(
    page_title="VA-Count Advanced Demo",
    page_icon="🔢",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}
.stage-header {
    font-size: 1.3rem;
    font-weight: bold;
    color: #1f77b4;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================
# MODEL LOADING (CACHED)
# ============================================


@st.cache_resource
def load_all_models():
    """Load all models (cached)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner("🔄 Loading models... This may take a minute..."):
        counting_model = load_counting_model("./data/checkpoint__finetuning_yolo.pth", device=device)
        yolo_model = load_yolo_model("./yolov8x-worldv2.pt", device=device)
        grounding_dino = load_grounding_dino(device=device)
        clip_model, clip_preprocess = load_clip_model(device=device)
        binary_classifier = load_binary_classifier(clip_model, device=device)
    st.success("✅ All models loaded successfully!")
    return {
        "counting_model": counting_model,
        "yolo_model": yolo_model,
        "grounding_dino": grounding_dino,
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "binary_classifier": binary_classifier,
        "device": device,
    }


@st.cache_data
def load_test_set_info():
    data_path = Path("./data/FSC147")
    image_dir = data_path / "images_384_VarV2"
    with open(data_path / "test.txt", "r") as f:
        all_test_images = [line.strip() for line in f.readlines()]
    test_images = [img for img in all_test_images if (image_dir / img).exists()]
    class_dict = {}
    with open(data_path / "ImageClasses_FSC147.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_name = parts[0]
                class_name = " ".join(parts[1:])
                class_dict[img_name] = class_name
    return test_images, class_dict


# ============================================
# HEADER
# ============================================

st.markdown(
    '<h1 class="main-header">🔢 VA-Count Advanced Demo</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Zero-Shot Object Counting with Good Exemplars [ECCV 2024] - Advanced Features</p>',
    unsafe_allow_html=True,
)

with st.expander("ℹ️ About this Advanced Demo"):
    st.markdown("""
    **VA-Count Advanced** adds:
    - **Expansion Prompt**: Use prompt expansion for both GroundingDINO and YOLOv8-World
    - **AI-Enhanced Prompts (NEW!)**: Use Gemini Vision to generate detailed visual descriptions
        - Example: "dog" → "single dog . brown fur . four legs . pointed ears . wagging tail ."
        - If object not in image, returns original prompt
        - Improves detection accuracy by providing specific visual cues
    - **Adjustable Binary Classifier Threshold**: Control confidence for single/multiple object patch
    """)

# ============================================
# SIDEBAR - SETTINGS
# ============================================

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("### 🔍 Detection")
    detection_model_type = st.selectbox(
        "Detection Model",
        ["GroundingDINO", "YOLOv8-World"],
        help="Choose detection model for object localization",
    )
    detection_threshold = st.slider(
        "Detection Threshold",
        min_value=0.01,
        max_value=0.50,
        value=0.05,
        step=0.01,
        help="Higher = fewer but more confident detections",
    )
    st.markdown("### ⭐ Exemplar Selection")
    top_k_pos = st.slider(
        "Positive Exemplars",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Number of positive exemplars (target objects)",
    )
    top_k_neg = st.slider(
        "Negative Exemplars",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        help="Number of negative exemplars (background/other objects)",
    )
    st.markdown("### 🎨 Visualization")
    overlay_alpha = st.slider(
        "Heatmap Overlay Intensity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0 = original image, 1 = heatmap only",
    )
    st.markdown("### 🧩 Expansion Prompt")
    use_expansion_prompt = st.checkbox(
        "Enable Expansion Prompt (for detection)",
        value=False,
        help="Use prompt expansion for detection stage",
    )
    st.markdown("### � AI Prompt Enhancement")
    use_ai_enhancement = st.checkbox(
        "Enable AI-Enhanced Prompts (Gemini Vision)",
        value=False,
        help="Use Gemini to generate detailed visual descriptions. If object not in image, returns original prompt.",
    )
    st.markdown("### �🧮 Binary Classifier Threshold")
    binary_classifier_threshold = st.slider(
        "Single Object Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.01,
        help="Confidence threshold for accepting a patch as single object",
    )

models = load_all_models()

# ============================================
# TABS
# ============================================

tab1, tab2 = st.tabs(["📸 Upload & Count", "🧪 Test Set Evaluation"])

# ============================================
# TAB 1: UPLOAD & COUNT
# ============================================

with tab1:
    st.markdown("## Upload Your Image")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload an image containing objects to count",
        )
    with col2:
        text_prompt = st.text_input(
            "Object to count",
            value="",
            placeholder="e.g., cow, apple, person",
            help="Enter the name of the object you want to count",
        )
    if uploaded_file is not None and text_prompt:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown(
            '<p class="stage-header">📸 Original Image</p>', unsafe_allow_html=True
        )
        st.image(image, use_column_width=True)
        if st.button("🚀 Count Objects!", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                # Start total inference timer
                total_start_time = time.time()
                
                # AI Prompt Enhancement logic
                ai_enhancement_time = 0.0
                if use_ai_enhancement:
                    with st.spinner("🤖 Enhancing prompt with AI..."):
                        try:
                            ai_start_time = time.time()
                            enhanced_prompt = enhance_prompt_with_gemini(image, text_prompt)
                            ai_enhancement_time = time.time() - ai_start_time
                            st.success(f"✨ Enhanced: `{enhanced_prompt}` (Time: {ai_enhancement_time:.3f}s)")
                            text_prompt_to_use = enhanced_prompt
                        except Exception as e:
                            ai_enhancement_time = time.time() - ai_start_time
                            st.warning(f"⚠ AI enhancement failed: {e}. Using simple prompt. (Time: {ai_enhancement_time:.3f}s)")
                            text_prompt_to_use = enhance_prompt_simple(text_prompt)
                else:
                    text_prompt_to_use = text_prompt
                
                # Expansion prompt logic
                if use_expansion_prompt:
                    expansion_prompts = expand_prompts(
                        text_prompt_to_use, enable_expansion=True
                    )
                    st.info(f"Expansion Prompts: {expansion_prompts}")
                    prompt_for_detection = expansion_prompts
                else:
                    prompt_for_detection = text_prompt_to_use
                # Run pipeline (pass threshold for binary classifier)
                results = advanced_counting_pipeline(
                    image=image,
                    text_prompt=text_prompt,
                    counting_model=models["counting_model"],
                    detection_model_type=detection_model_type,
                    grounding_dino_model=models["grounding_dino"],
                    yolo_model=models["yolo_model"],
                    clip_model=models["clip_model"],
                    clip_preprocess=models["clip_preprocess"],
                    binary_classifier=models["binary_classifier"],
                    device=models["device"],
                    detection_threshold=detection_threshold,
                    top_k_pos=top_k_pos,
                    top_k_neg=top_k_neg,
                    prompt_for_detection=prompt_for_detection,
                    binary_classifier_threshold=binary_classifier_threshold,
                )
                
                # Calculate total inference time
                total_inference_time = time.time() - total_start_time
            st.markdown("---")
            
            # Display Total Inference Time at the top
            st.markdown(
                f"""
            <div class="metric-box" style="margin-bottom: 1.5rem;">
                <h2 style="color: #1f77b4; margin: 0;">⏱️ Total Inference Time: {total_inference_time:.3f}s</h2>
                <p style="color: #666; margin: 0;">
                    AI Enhancement: {ai_enhancement_time:.3f}s | Detection: {results['detection_time']:.3f}s | 
                    Classifier: {results.get('classifier_time', 0):.3f}s | Counting: {results.get('counting_time', 0):.3f}s
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            
            st.markdown("## 📊 Results")

            # Stage 1: Raw Detections
            st.markdown(
                f'<p class="stage-header">🔍 Stage 1: Raw Detections ({detection_model_type})</p>',
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if len(results["raw_boxes"]) > 0:
                    vis_img = visualize_detection_stage(
                        image, results["raw_boxes"], results["raw_scores"]
                    )
                    st.image(vis_img, use_column_width=True)
                else:
                    st.image(image, use_column_width=True)
                    st.error("❌ No objects detected")
            with col2:
                st.metric("Total Detected", len(results["raw_boxes"]))
                st.metric("Detection Time", f"{results['detection_time']:.3f}s")
                if use_ai_enhancement:
                    st.metric("AI Enhancement Time", f"{ai_enhancement_time:.3f}s")
            with col3:
                st.caption("**Detection Scores:**")
                for i, score in enumerate(results["raw_scores"][:5]):
                    st.caption(f"Box {i + 1}: {score:.3f}")
                if len(results["raw_scores"]) > 5:
                    st.caption(f"... and {len(results['raw_scores']) - 5} more")

            # Stage 2: Binary Classifier
            if len(results["raw_boxes"]) > 0:
                st.markdown(
                    '<p class="stage-header">✅ Stage 2: Binary Classifier (Single Object Filter)</p>',
                    unsafe_allow_html=True,
                )
                col1, col2 = st.columns([2, 1])
                with col1:
                    vis_img = visualize_classifier_stage(
                        image,
                        results["accepted_boxes"],
                        results["rejected_boxes"],
                        results["classifier_scores"],
                    )
                    st.image(vis_img, use_column_width=True)
                with col2:
                    st.metric("✅ Accepted (Single)", len(results["accepted_boxes"]))
                    st.metric("❌ Rejected (Multiple)", len(results["rejected_boxes"]))

            # Stage 3: Selected Exemplars
            if len(results["pos_boxes"]) > 0:
                st.markdown(
                    '<p class="stage-header">⭐ Stage 3: Selected Exemplars</p>',
                    unsafe_allow_html=True,
                )
                vis_img = visualize_exemplars(
                    image,
                    results["pos_boxes"],
                    results["pos_scores"],
                    results.get("neg_boxes", np.array([])),
                    results.get("neg_scores", []),
                )
                st.image(vis_img, use_column_width=True)
                pos_crops = extract_crops_for_display(
                    image, results["pos_boxes"], max_crops=3
                )
                if len(results.get("neg_boxes", [])) > 0:
                    neg_crops = extract_crops_for_display(
                        image, results["neg_boxes"], max_crops=3
                    )
                    all_crops = pos_crops + neg_crops
                    cols = st.columns(len(all_crops))
                    for i, (crop, col) in enumerate(zip(all_crops, cols)):
                        with col:
                            st.image(
                                crop,
                                caption=f"Exemplar {i + 1}",
                                use_column_width=True,
                            )
                else:
                    cols = st.columns(len(pos_crops))
                    for i, (crop, col) in enumerate(zip(pos_crops, cols)):
                        with col:
                            st.image(
                                crop,
                                caption=f"Positive {i + 1}",
                                use_column_width=True,
                            )

            # Stage 4: Density Map & Count
            st.markdown(
                '<p class="stage-header">🔥 Stage 4: Density Map & Final Count</p>',
                unsafe_allow_html=True,
            )
            
            if results["density_map"] is not None:
                st.markdown(
                    f"""
                <div class="metric-box">
                    <h1 style="color: #1f77b4; margin: 0;">Count: {int(round(results["count"]))}</h1>
                    <p style="color: #666; margin: 0;">Exact: {results["count"]:.2f}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Original Image**")
                    st.image(results["processed_image"], use_column_width=True)
                with col2:
                    st.markdown("**Heatmap Only**")
                    heatmap = create_density_heatmap(results["density_map"])
                    st.image(heatmap, use_column_width=True)
                with col3:
                    st.markdown(f"**Overlay (α={overlay_alpha})**")
                    overlay = create_overlay_image(
                        results["processed_image"],
                        results["density_map"],
                        alpha=overlay_alpha,
                    )
                    st.image(overlay, use_column_width=True)
                st.success("✅ Counting completed!")
    elif uploaded_file is None:
        st.info("Upload an image to get started")
    elif not text_prompt:
        st.info("Enter the object name to count")

# ============================================
# TAB 2: TEST SET EVALUATION
# ============================================

with tab2:
    st.markdown("## Test Set Evaluation")
    st.caption("Compare model predictions with ground truth on FSC147 test set")
    test_images, class_dict = load_test_set_info()
    selected_image = st.selectbox(
        "Select Test Image",
        test_images,
        help="Choose an image from the FSC147 test set",
    )
    if selected_image:
        class_name = class_dict.get(selected_image, "unknown")
        data_path = Path("./data/FSC147")
        image_path = data_path / "images_384_VarV2" / selected_image
        gt_path = (
            data_path
            / "gt_density_map_adaptive_384_VarV2"
            / selected_image.replace(".jpg", ".npy")
        )
        if not image_path.exists():
            st.error(f"Image not found: {image_path}")
        else:
            image = Image.open(image_path).convert("RGB")
            gt_density = np.load(gt_path) if gt_path.exists() else None
            gt_count = (gt_density.sum() / 60) if gt_density is not None else 0
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Image", selected_image)
            with col2:
                st.metric("Class", class_name)
            with col3:
                st.metric("GT Count", f"{int(round(gt_count))}")
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📸 Test Image**")
                st.image(image, use_column_width=True)
            with col2:
                if gt_density is not None:
                    st.markdown("**🎯 Ground Truth Density**")
                    gt_vis = visualize_gt_density(gt_density)
                    st.image(gt_vis, use_column_width=True)
            if st.button("🚀 Run Inference", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    if use_expansion_prompt:
                        expansion_prompts = expand_prompts(
                            class_name, enable_expansion=True
                        )
                        st.info(f"Expansion Prompts: {expansion_prompts}")
                        prompt_for_detection = expansion_prompts
                    else:
                        prompt_for_detection = class_name
                    results = advanced_counting_pipeline(
                        image=image,
                        text_prompt=class_name,
                        counting_model=models["counting_model"],
                        detection_model_type=detection_model_type,
                        grounding_dino_model=models["grounding_dino"],
                        yolo_model=models["yolo_model"],
                        clip_model=models["clip_model"],
                        clip_preprocess=models["clip_preprocess"],
                        binary_classifier=models["binary_classifier"],
                        device=models["device"],
                        detection_threshold=detection_threshold,
                        top_k_pos=top_k_pos,
                        top_k_neg=top_k_neg,
                        prompt_for_detection=prompt_for_detection,
                        binary_classifier_threshold=binary_classifier_threshold,
                    )
                st.markdown("---")
                st.markdown("## 📊 Pipeline Results")
                # ... (reuse visualization code from demo_app.py as needed) ...

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <p><b>VA-Count Advanced</b> - Zero-shot Object Counting with Good Exemplars</p>
    <p><i>ECCV 2024</i></p>
</div>
""",
    unsafe_allow_html=True,
)
