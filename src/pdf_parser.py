"""
pdf_parser.py
-------------
Multi-layer PDF parsing:

  Layer 1 (PyMuPDF / fitz)   -> headings, paragraphs, images, layout positions
  Layer 2 (pdfplumber)       -> fallback text + simple tables
  Layer 3 (Camelot)          -> high-accuracy table extraction (lattice + stream)
  Layer 4 (BLIP captioning)  -> auto-caption every extracted image

Design choice: we don't rely on ONE library because none of them is
complete on its own —
  - PyMuPDF is fast and gives font-size info (great for heading detection)
    but its table detection is weak.
  - pdfplumber gives clean word-level text + basic tables.
  - Camelot is the most accurate table extractor but needs Ghostscript
    and only works well on ruled / well-spaced tables.
  - unstructured.io is used as a high-level fallback for messy PDFs
    (scanned, irregular layouts) where the above three struggle.
"""

import fitz  
import pdfplumber
import re
import os
from typing import List, Optional
from schema import DocElement, ElementType, BoundingBox, ParsedDocument
from captioner import caption_image


# ─────────────────────────────────────────────
# HEADING DETECTION HEURISTIC
# ─────────────────────────────────────────────
def classify_heading_level(font_size: float, body_font_size: float, is_bold: bool) -> Optional[int]:
    """
    Heuristic heading detector based on relative font size + boldness.
    Real-world PDFs don't carry <h1>/<h2> tags — we infer structure
    from typography, same approach used by unstructured.io internally.
    """
    ratio = font_size / max(body_font_size, 1)
    if ratio >= 1.8:
        return 1
    elif ratio >= 1.5:
        return 2
    elif ratio >= 1.25 and is_bold:
        return 3
    elif ratio >= 1.1 and is_bold:
        return 4
    return None


def get_body_font_size(doc: fitz.Document) -> float:
    """Find the most common font size in the doc -> treat as 'body text'."""
    sizes = {}
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    sz = round(span["size"], 1)
                    sizes[sz] = sizes.get(sz, 0) + len(span["text"])
    if not sizes:
        return 11.0
    return max(sizes, key=sizes.get)


# ─────────────────────────────────────────────
# TEXT + HEADING EXTRACTION (PyMuPDF)
# ─────────────────────────────────────────────
def extract_text_elements(doc: fitz.Document, doc_id: str) -> List[DocElement]:
    elements = []
    body_size = get_body_font_size(doc)
    elem_counter = 0

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:   # 0 = text block, 1 = image block
                continue

            block_text_parts = []
            max_size = 0
            any_bold = False
            bbox = block["bbox"]

            for line in block["lines"]:
                line_text = "".join(span["text"] for span in line["spans"])
                block_text_parts.append(line_text)
                for span in line["spans"]:
                    max_size = max(max_size, span["size"])
                    if "bold" in span["font"].lower():
                        any_bold = True

            text = "\n".join(block_text_parts).strip()
            if not text:
                continue

            heading_level = classify_heading_level(max_size, body_size, any_bold)

            # Detect bullet/numbered list items
            is_list = bool(re.match(r"^\s*([\u2022\-\*]|\d+[\.\)])\s+", text))

            etype = (
                ElementType.HEADING if heading_level
                else ElementType.LIST_ITEM if is_list
                else ElementType.PARAGRAPH
            )

          

            elements.append(DocElement(
                element_id=f"{doc_id}_p{page_num}_el{elem_counter}",
                type=etype,
                content=text,
                page=page_num,
                position=BoundingBox(*bbox),
                heading_level=heading_level,
                metadata={"font_size": round(max_size, 1), "bold": any_bold},
            ))
            elem_counter+=1

    return elements


# ─────────────────────────────────────────────
# IMAGE EXTRACTION + CAPTIONING (PyMuPDF + BLIP)
# ─────────────────────────────────────────────
def extract_image_elements(doc: fitz.Document, doc_id: str, output_dir: str) -> List[DocElement]:
    elements = []
    os.makedirs(output_dir, exist_ok=True)
    elem_counter = 0

    for page_num, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
            except Exception:
                continue

            elem_counter += 1
            img_filename = f"{doc_id}_p{page_num}_img{img_index}.{ext}"
            img_path = os.path.join(output_dir, img_filename)
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            # Get image bbox on page (if available)
            try:
                rects = page.get_image_rects(xref)
                bbox = rects[0] if rects else None
            except Exception:
                bbox = None

            # Auto-caption using lightweight vision model (BLIP)
            caption = caption_image(img_path)

            elements.append(DocElement(
                element_id=f"{doc_id}_p{page_num}_el{elem_counter}_img",
                type=ElementType.IMAGE,
                content=caption,
                page=page_num,
                position=BoundingBox(bbox.x0, bbox.y0, bbox.x1, bbox.y1) if bbox else None,
                metadata={"image_path": img_path, "format": ext, "auto_captioned": True},
            ))

    return elements


# ─────────────────────────────────────────────
# TABLE EXTRACTION (pdfplumber primary, Camelot optional)
# ─────────────────────────────────────────────
def table_to_markdown(table_data: List[List[Optional[str]]]) -> str:
    """Convert a raw table (list of rows) into clean markdown."""
    if not table_data or not table_data[0]:
        return ""
    rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in table_data]
    header = rows[0]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows[1:]:
        # pad row to header length
        row = row + [""] * (len(header) - len(row))
        md += "| " + " | ".join(row[:len(header)]) + " |\n"
    return md.strip()


def extract_table_elements(pdf_path: str, doc_id: str) -> List[DocElement]:
    elements = []
    elem_counter = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            for t_idx, table in enumerate(tables):
                data = table.extract()
                if not data or len(data) < 2:
                    continue

                elem_counter += 1
                md_table = table_to_markdown(data)
                bbox = table.bbox  # (x0, top, x1, bottom)

                elements.append(DocElement(
                    element_id=f"{doc_id}_p{page_num}_el{elem_counter}_table",
                    type=ElementType.TABLE,
                    content=md_table,
                    page=page_num,
                    position=BoundingBox(*bbox),
                    metadata={
                        "rows": len(data),
                        "cols": len(data[0]) if data else 0,
                        "extraction_method": "pdfplumber",
                    },
                ))

    return elements


def extract_table_elements_camelot(pdf_path: str, doc_id: str) -> List[DocElement]:
    """
    Higher-accuracy table extraction using Camelot.
    Falls back silently if Camelot/Ghostscript isn't available —
    pdfplumber tables above already provide baseline coverage.
    """
    elements = []
    try:
        import camelot
    except ImportError:
        return elements

    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        for i, table in enumerate(tables):
            md_table = table_to_markdown(table.df.values.tolist())
            elements.append(DocElement(
                element_id=f"{doc_id}_camelot_t{i}",
                type=ElementType.TABLE,
                content=md_table,
                page=int(table.page),
                metadata={
                    "extraction_method": "camelot_lattice",
                    "accuracy": float(table.parsing_report.get("accuracy", 0)),
                },
            ))
    except Exception as e:
        print(f"[!] Camelot extraction skipped: {e}")

    return elements


# ─────────────────────────────────────────────
# CAPTION ATTACHMENT (link "Figure 1: ..." text to nearest image)
# ─────────────────────────────────────────────
def attach_captions(elements: List[DocElement]) -> List[DocElement]:
    """
    Detect paragraph elements that look like figure/table captions
    (e.g. "Figure 1: ...", "Table 2. ...") and re-tag them as CAPTION
    type, linking to the nearest image/table on the same page.
    """
    caption_pattern = re.compile(r"^(Figure|Fig\.?|Table)\s+\d+[:.]\s*", re.IGNORECASE)

    for el in elements:
        if el.type == ElementType.PARAGRAPH and caption_pattern.match(el.content):
            el.type = ElementType.CAPTION
            # find nearest image/table on same page
            candidates = [
                e for e in elements
                if e.page == el.page and e.type in (ElementType.IMAGE, ElementType.TABLE)
            ]
            if candidates and el.position:
                nearest = min(
                    candidates,
                    key=lambda c: abs((c.position.y0 if c.position else 0) - el.position.y0)
                )
                el.metadata["linked_element_id"] = nearest.element_id
                nearest.metadata["caption_element_id"] = el.element_id

    return elements


# ─────────────────────────────────────────────
# MAIN PARSE FUNCTION
# ─────────────────────────────────────────────
def parse_pdf(pdf_path: str, output_dir: str = "./output/images", use_camelot: bool = False) -> ParsedDocument:
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)

    print(f"[*] Parsing PDF: {pdf_path} ({len(doc)} pages)")

    text_elements = extract_text_elements(doc, doc_id)
    print(f"    -> {len(text_elements)} text/heading/list elements")

    image_elements = extract_image_elements(doc, doc_id, output_dir)
    print(f"    -> {len(image_elements)} images (captioned)")

    table_elements = extract_table_elements(pdf_path, doc_id)
    print(f"    -> {len(table_elements)} tables (pdfplumber)")

    if use_camelot:
        camelot_tables = extract_table_elements_camelot(pdf_path, doc_id)
        table_elements.extend(camelot_tables)
        print(f"    -> +{len(camelot_tables)} tables (camelot)")

    all_elements = text_elements + image_elements + table_elements
    all_elements = attach_captions(all_elements)

    # Sort by page, then vertical position
    all_elements.sort(key=lambda e: (e.page, e.position.y0 if e.position else 0))

    # --- HIERARCHY TAGGING ---
    current_parents = {} # level -> id
    for el in all_elements:
        if el.type == ElementType.HEADING:
            lvl = el.heading_level or 1
            current_parents[lvl] = el.element_id
            # clear higher levels
            for l in range(lvl + 1, 7):
                current_parents.pop(l, None)
            
            # find immediate parent (level - 1)
            for l in range(lvl - 1, 0, -1):
                if l in current_parents:
                    el.parent_id = current_parents[l]
                    break
        else:
            # find deepest current heading
            if current_parents:
                deepest_lvl = max(current_parents.keys())
                el.parent_id = current_parents[deepest_lvl]
    # -------------------------

    parsed = ParsedDocument(
        source_file=pdf_path,
        doc_type="pdf",
        total_pages=len(doc),
        elements=all_elements,
        parse_stats={
            "total_elements": len(all_elements),
            "by_type": {},
        },
    )
    parsed.parse_stats["by_type"] = parsed.summary()
    doc.close()
    return parsed


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    result = parse_pdf(sys.argv[1])
    print(f"\n[✓] Parsed {result.parse_stats['total_elements']} elements")
    print(f"[✓] Breakdown: {result.parse_stats['by_type']}")
    result.save("./output/parsed_output.json")
    print("[✓] Saved to ./output/parsed_output.json")
