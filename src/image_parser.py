"""
image_parser.py
---------------
Standalone image parsing (PNG, JPG, JPEG).
Wraps the image in a ParsedDocument and uses BLIP for auto-captioning.
"""

import os
from PIL import Image
from typing import List
from schema import DocElement, ElementType, ParsedDocument
from captioner import caption_image


def parse_image(file_path: str, output_dir: str = "./output/images") -> ParsedDocument:
    """
    Parses a standalone image file, generates a caption, and returns a ParsedDocument.
    """
    doc_id = os.path.splitext(os.path.basename(file_path))[0]
    ext = os.path.splitext(file_path)[1].lower().strip(".")
    
    print(f"[*] Parsing Image: {file_path}")
    
    # Ensure captioner output dir concept is consistent (though we use original file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate auto-caption
    caption, confidence = caption_image(file_path)
    
    # Get image dimensions
    try:
        with Image.open(file_path) as img:
            width, height = img.size
    except Exception:
        width, height = 0, 0

    # Create the single element
    element = DocElement(
        element_id=f"{doc_id}_img_el1",
        type=ElementType.IMAGE,
        content=caption,
        page=1,
        confidence=confidence,
        metadata={
            "image_path": file_path,
            "format": ext,
            "width": width,
            "height": height,
            "auto_captioned": True
        }
    )

    parsed = ParsedDocument(
        source_file=file_path,
        doc_type="image",
        total_pages=1,
        elements=[element],
        parse_stats={
            "total_elements": 1,
            "by_type": {"image": 1},
        },
    )
    return parsed


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python image_parser.py <path_to_image>")
        sys.exit(1)

    try:
        result = parse_image(sys.argv[1])
        print(f"\n[✓] Parsed image: {result.source_file}")
        print(f"    Caption: {result.elements[0].content}")
        
        # Save sample
        os.makedirs("./output", exist_ok=True)
        out_path = "./output/sample_image_parsed.json"
        result.save(out_path)
        print(f"[✓] Saved results to {out_path}")
    except Exception as e:
        print(f"[✗] Failed: {e}")
