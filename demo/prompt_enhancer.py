"""Prompt Enhancement using the Gemini API.

Enhanced prompts (rich, visual definitions) improve detection and counting
accuracy of GroundingDINO / YOLO-World on FSC147.

The API key is read from the ``GEMINI_API_KEY`` environment variable.
For local development you can put it in a ``.env`` file at
``code/source-code/.env`` (see ``.env.example`` for the format) -- the file is
loaded automatically via ``python-dotenv``. The key MUST NOT be committed to
the repository.
"""

import os
import time

import google.generativeai as genai
import inflect
from PIL import Image

try:
    from dotenv import load_dotenv

    # Load `.env` if present, but do not override variables already exported
    # in the current shell.
    load_dotenv(override=False)
except ImportError:  # pragma: no cover - python-dotenv is an optional helper
    # `python-dotenv` is not installed: fall back to plain os.environ. The user
    # can still export GEMINI_API_KEY manually before running.
    pass

# Initialize
p = inflect.engine()

# Read the API key from the environment. Do NOT hard-code secrets.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Export it in your shell or place it in "
        "`code/source-code/.env` (see `.env.example`). "
        "Never commit the key to the repository."
    )

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)


def enhance_prompt_with_gemini(image, class_name, max_retries=2):
    """
    Enhance prompt using Gemini Vision API
    
    Args:
        image: PIL Image
        class_name: Base class name (e.g., "dog", "car")
        max_retries: Number of retry attempts
    
    Returns:
        Enhanced prompt string or original class_name if fails
    """
    # 1. Singularize
    singular_name = p.singular_noun(class_name)
    if not singular_name:
        singular_name = class_name
    
    # 2. Craft prompt for "Visual Definition"
    prompt = f"""
Look at the image and provide the **visual definition** of a single '{singular_name}'.

**CRITICAL RULES:**
1. **IF the image does NOT contain any '{singular_name}'**, respond with ONLY: "single {singular_name} ."
2. **IF the image DOES contain '{singular_name}'**, describe ONE instance's visual appearance.

**Task**: Describe the intrinsic physical appearance of just **ONE** instance, as if it were cropped out and isolated.

**Format Rules:**
- Start with 'single {singular_name}'
- Use dot-separated phrases (e.g., "single dog . brown fur . four legs .")
- Focus on: Shape, Color, Material, Texture
- Ignore background or other objects
- End with a dot

**Example for 'keyboard key':** 
BAD: keyboard key . rows of buttons . full keyboard layout .
GOOD: single keyboard key . square shape . black plastic material . white printed letter . smooth surface .

**Your output for '{singular_name}':**
"""
    
    # 3. Call Gemini API with retry
    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, image])
            text = response.text.strip().replace("\n", " ").replace("..", ".")
            
            # Ensure ends with dot
            if not text.endswith('.'):
                text += ' .'
            
            # Validate response
            if text and len(text) > 5:
                return text
            else:
                # Empty/invalid response, return fallback
                return f"single {singular_name} ."
                
        except Exception as e:
            print(f"⚠ Gemini API error (attempt {attempt+1}/{max_retries}): {e}")
            
            # Handle rate limit
            if "429" in str(e) or "quota" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
            
            # Handle "object not found" gracefully
            if attempt == max_retries - 1:
                return f"single {singular_name} ."
    
    # Fallback
    return f"single {singular_name} ."


def enhance_prompt_simple(class_name):
    """
    Simple prompt enhancement without Gemini (fallback)
    Just singularizes and formats
    """
    singular_name = p.singular_noun(class_name)
    if not singular_name:
        singular_name = class_name
    
    return f"single {singular_name} ."


# Test function
if __name__ == "__main__":
    # Test with sample image
    test_image_path = "./data/FSC147/images_384_VarV2/2.jpg"
    test_class = "dog"
    
    try:
        img = Image.open(test_image_path)
        enhanced = enhance_prompt_with_gemini(img, test_class)
        print(f"Original: {test_class}")
        print(f"Enhanced: {enhanced}")
    except Exception as e:
        print(f"Test failed: {e}")
