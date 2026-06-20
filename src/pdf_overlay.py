"""
pdf_overlay.py  ─  Visualize relationships directly on the source PDF
------------------------------------------------------------------------
Draws colored bounding boxes around every parsed element and connecting
lines for the "interesting" relationship types (caption↔figure/table,
cross-references, footnotes, semantic co-references) — the relationships
that are NOT obvious just from reading the page top-to-bottom.

This is the strongest "wow" visual for a hackathon demo: it makes the
graph's value tangible by showing exactly which paragraph references
which table, right on the original document.

Output:
  - An annotated copy of the PDF (overlay_<name>.pdf)
  - One PNG per page (for dropping straight into slides)

Usage
-----
    python pdf_overlay.py <source.pdf> <doc_graph.pkl> [output_dir]
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import fitz  # PyMuPDF

from graph import DocumentGraph
from schema import ElementType, EdgeType


# ── Styling (fitz uses RGB in 0–1 range, not 0–255) ─────────────────────────

def _rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))


NODE_BOX_COLORS = {
    ElementType.HEADING:   _rgb("#E63946"),
    ElementType.PARAGRAPH: _rgb("#457B9D"),
    ElementType.TABLE:     _rgb("#2A9D8F"),
    ElementType.IMAGE:     _rgb("#F4A261"),
    ElementType.CAPTION:   _rgb("#E9C46A"),
    ElementType.LIST_ITEM: _rgb("#A8DADC"),
}

# Only draw lines for relationship types that aren't visually obvious from
# reading order — sequential/proximity/same_section would clutter the page
# since they connect almost everything. These are the "discoveries."
EDGE_LINE_COLORS = {
    EdgeType.CAPTION_OF:   _rgb("#F4A261"),
    EdgeType.CROSS_REF:    _rgb("#FFB703"),
    EdgeType.FOOTNOTE_REF: _rgb("#6A4C93"),
    EdgeType.CO_REFERENCE: _rgb("#2A9D8F"),
}

DEFAULT_EDGE_TYPES = list(EDGE_LINE_COLORS.keys())


def _safe_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return " ".join(str(x) for x in content if x is not None)
    return str(content) if content is not None else ""


def _bbox_center(bbox) -> fitz.Point:
    return fitz.Point((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)


# ── Core overlay logic ───────────────────────────────────────────────────────

def draw_overlay(
    pdf_path: str,
    graph: DocumentGraph,
    output_dir: str,
    edge_types: list[EdgeType] = None,
) -> tuple[str, list[str]]:
    edge_types = edge_types or DEFAULT_EDGE_TYPES
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    # ── Pass 1: bounding boxes for every positioned element ─────────────────
    for node in graph.all_nodes():
        if not node.position:
            continue
        page_idx = node.page - 1
        if page_idx < 0 or page_idx >= len(doc):
            continue
        page = doc[page_idx]
        rect = fitz.Rect(node.position.x0, node.position.y0,
                         node.position.x1, node.position.y1)
        color = NODE_BOX_COLORS.get(node.type, (0.6, 0.6, 0.6))
        page.draw_rect(rect, color=color, width=1.2, overlay=True)

    # ── Pass 2: relationship lines (deduped, same-page only) ────────────────
    seen_pairs: set[tuple[str, str, str]] = set()
    edge_counts: dict[str, int] = {}

    for src_id, tgt_id, data in graph._g.edges(data=True):
        et = data.get("edge_type")
        if et not in edge_types:
            continue

        key = (src_id, tgt_id, et.value)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        src = graph.get_node(src_id)
        tgt = graph.get_node(tgt_id)
        if not src or not tgt or not src.position or not tgt.position:
            continue
        if src.page != tgt.page:
            continue  # only draw what's visible on a single page

        page_idx = src.page - 1
        if page_idx < 0 or page_idx >= len(doc):
            continue
        page = doc[page_idx]

        p1 = _bbox_center(src.position)
        p2 = _bbox_center(tgt.position)
        color = EDGE_LINE_COLORS.get(et, (0.5, 0.5, 0.5))

        page.draw_line(p1, p2, color=color, width=1.4, dashes="[2 2] 0")
        page.draw_circle(p1, 2.5, color=color, fill=color)
        page.draw_circle(p2, 2.5, color=color, fill=color)

        edge_counts[et.value] = edge_counts.get(et.value, 0) + 1

    # ── Pass 3: legend on page 1 ─────────────────────────────────────────────
    if len(doc) > 0:
        _draw_legend(doc[0], edge_types, edge_counts)

    # ── Save annotated PDF ────────────────────────────────────────────────────
    out_pdf = out_dir / f"{doc_id}_overlay.pdf"
    doc.save(str(out_pdf))
    print(f"[Overlay] Saved annotated PDF → {out_pdf}")

    # ── Save per-page PNGs (for slides) ───────────────────────────────────────
    png_paths = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=150)
        png_path = out_dir / f"{doc_id}_overlay_p{i}.png"
        pix.save(str(png_path))
        png_paths.append(str(png_path))
        print(f"[Overlay] Saved page image → {png_path}")

    doc.close()
    print(f"[Overlay] Relationship lines drawn: {edge_counts}")
    return str(out_pdf), png_paths


def _draw_legend(page: fitz.Page, edge_types: list[EdgeType], edge_counts: dict[str, int]) -> None:
    """Small color-key box in the top-right corner of page 1."""
    x0, y0 = page.rect.width - 170, 10
    line_h = 14
    box_h = line_h * (len(edge_types) + 1) + 10

    page.draw_rect(
        fitz.Rect(x0 - 5, y0 - 5, page.rect.width - 5, y0 + box_h),
        color=(0.2, 0.2, 0.2), fill=(1, 1, 1), fill_opacity=0.85, width=0.5,
    )
    page.insert_text(fitz.Point(x0, y0 + 10), "Relationship Key", fontsize=8, color=(0, 0, 0))

    for i, et in enumerate(edge_types, start=1):
        y = y0 + 10 + i * line_h
        color = EDGE_LINE_COLORS.get(et, (0.5, 0.5, 0.5))
        count = edge_counts.get(et.value, 0)
        page.draw_line(fitz.Point(x0, y - 3), fitz.Point(x0 + 15, y - 3), color=color, width=2)
        page.insert_text(
            fitz.Point(x0 + 20, y), f"{et.value} ({count})", fontsize=7, color=(0, 0, 0)
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pdf_overlay.py <source.pdf> <doc_graph.pkl> [output_dir]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    pkl_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "../output/images"

    graph = DocumentGraph.load(pkl_path)
    print(f"[Overlay] Loaded graph: {graph}")
    draw_overlay(pdf_path, graph, output_dir)