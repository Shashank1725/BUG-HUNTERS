"""
captioner.py
------------
Lightweight image-to-caption conversion using BLIP (Salesforce).

Why BLIP and not GPT-4V / a bigger VLM?
  - BLIP-base is ~990MB, runs on CPU in <1s per image
  - No API cost, no network dependency — works fully offline
  - Good enough quality for "what is this figure showing" captions,
    which is the use-case here (not fine-grained VQA)

Falls back to a lightweight heuristic captioner (size/format based)
if `transformers`/`torch` aren't available, so the pipeline never
hard-crashes on a missing dependency during a live demo.
"""

import os
import warnings
# Silence all warnings for a clean user experience
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from typing import Optional

_blip_model = None
_blip_processor = None
_blip_available = None


def _try_load_blip():
    """Lazy-load BLIP only once, only if actually needed."""
    global _blip_model, _blip_processor, _blip_available
    if _blip_available is not None:
        return _blip_available

    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration, logging

        # Silence transformers internal logging
        logging.set_verbosity_error()

        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_available = True
        print("[*] BLIP captioning model loaded")
    except Exception as e:
        print(f"[!] BLIP unavailable ({e}); using fallback captioner")
        _blip_available = False

    return _blip_available


def _fallback_caption(image_path: str) -> str:
    """
    Heuristic caption when no vision model is available.
    Still gives SOME useful metadata rather than an empty string.
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        aspect = "wide" if w > h * 1.3 else "tall" if h > w * 1.3 else "square"
        return f"[Image: {aspect} graphic, {w}x{h}px — caption unavailable, vision model not loaded]"
    except Exception:
        return "[Image: unable to read file for captioning]"


def caption_image(image_path: str, max_length: int = 40) -> str:
    """
    Generate a natural-language caption for an image file.
    Used by both the PDF and DOCX parsers for every extracted image.
    """
    if not os.path.exists(image_path):
        return "[Image: file not found]"

    if not _try_load_blip():
        return _fallback_caption(image_path)

    try:
        from PIL import Image
        raw_image = Image.open(image_path).convert("RGB")
        inputs = _blip_processor(raw_image, return_tensors="pt")
        out = _blip_model.generate(**inputs, max_length=max_length)
        caption = _blip_processor.decode(out[0], skip_special_tokens=True)
        return caption.strip().capitalize()
    except Exception as e:
        print(f"[!] Captioning failed for {image_path}: {e}")
        return _fallback_caption(image_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python captioner.py <path_to_image>")
        sys.exit(1)
    print(caption_image(sys.argv[1]))
