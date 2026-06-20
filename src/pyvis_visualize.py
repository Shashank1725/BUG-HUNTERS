"""
pyvis_visualize.py  ─  Interactive HTML graph visualization
-------------------------------------------------------------
Builds a zoomable, pannable, drag-able interactive graph using PyVis.
Designed for hackathon demos — judges can click nodes, see content,
drag to explore relationships, and filter visually by color.

Usage
-----
    python pyvis_visualize.py ../output/doc_graph.pkl ../output/images/graph_interactive.html
"""

from __future__ import annotations

import sys
from pathlib import Path

from pyvis.network import Network

from graph import DocumentGraph
from schema import ElementType, EdgeType


# ── Visual styling ──────────────────────────────────────────────────────────

NODE_COLORS = {
    ElementType.HEADING:   "#E63946",  # red
    ElementType.PARAGRAPH: "#457B9D",  # blue
    ElementType.TABLE:     "#2A9D8F",  # teal
    ElementType.IMAGE:     "#F4A261",  # orange
    ElementType.CAPTION:   "#E9C46A",  # yellow
    ElementType.LIST_ITEM: "#A8DADC",  # light blue
}

NODE_SHAPES = {
    ElementType.HEADING:   "box",
    ElementType.PARAGRAPH: "dot",
    ElementType.TABLE:     "square",
    ElementType.IMAGE:     "image" if False else "triangle",  # 'image' shape needs a url; use triangle
    ElementType.CAPTION:   "ellipse",
    ElementType.LIST_ITEM: "dot",
}

EDGE_COLORS = {
    EdgeType.SEQUENTIAL:   "#CCCCCC",
    EdgeType.PARENT_CHILD: "#E63946",
    EdgeType.SAME_SECTION: "#A8DADC",
    EdgeType.CAPTION_OF:   "#F4A261",
    EdgeType.FOOTNOTE_REF: "#6A4C93",
    EdgeType.CROSS_REF:    "#FFB703",
    EdgeType.CO_REFERENCE: "#2A9D8F",
    EdgeType.PROXIMITY:    "#DDDDDD",
}

EDGE_WIDTH = {
    EdgeType.PARENT_CHILD: 3,
    EdgeType.CAPTION_OF:   3,
    EdgeType.CROSS_REF:    2,
    EdgeType.FOOTNOTE_REF: 2,
    EdgeType.CO_REFERENCE: 2,
    EdgeType.SAME_SECTION: 1,
    EdgeType.SEQUENTIAL:   1,
    EdgeType.PROXIMITY:    1,
}


# def _truncate(text: str, limit: int = 120) -> str:
#     text = (text or "").strip().replace("\n", " ")
#     return text if len(text) <= limit else text[:limit] + "…"
def _safe_content(content) -> str:
    """Guards against parsers that store tuples, lists, or None in content."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return " ".join(str(x) for x in content if x is not None)
    return str(content) if content is not None else ""


def _truncate(text, limit: int = 120) -> str:
    text = _safe_content(text).strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def build_interactive_graph(graph: DocumentGraph, out_path: str) -> None:
    net = Network(
        height="850px",
        width="100%",
        bgcolor="#0F1116",
        font_color="white",
        directed=True,
        notebook=False,
    )

    # Physics: stable, demo-friendly layout that still allows drag/zoom/pan
    net.barnes_hut(
        gravity=-2500,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.04,
        damping=0.9,
    )

    # ── Add nodes ────────────────────────────────────────────────────────────
    for node in graph.all_nodes():
        color = NODE_COLORS.get(node.type, "#999999")
        shape = NODE_SHAPES.get(node.type, "dot")
        preview = _truncate(node.content, 200)

        label = f"{node.type.value.upper()} (p{node.page})"
        title = (
            f"<b>{node.element_id}</b><br>"
            f"Type: {node.type.value}<br>"
            f"Page: {node.page}<br><br>"
            f"{preview}"
        )

        size = 25 if node.type in (ElementType.HEADING, ElementType.TABLE, ElementType.IMAGE) else 15

        net.add_node(
            node.element_id,
            label=label,
            title=title,
            color=color,
            shape=shape,
            size=size,
        )

    # ── Add edges ────────────────────────────────────────────────────────────
    seen_edges = set()
    for src, tgt, data in graph._g.edges(data=True):
        et = data.get("edge_type")
        et_val = et.value if hasattr(et, "value") else str(et)
        key = (src, tgt, et_val)
        if key in seen_edges:
            continue
        seen_edges.add(key)

        color = EDGE_COLORS.get(et, "#888888")
        width = EDGE_WIDTH.get(et, 1)
        weight = data.get("weight", 1.0)

        net.add_edge(
            src, tgt,
            color=color,
            width=width,
            title=f"{et_val} (w={weight:.2f})",
            arrows="to",
        )

    # ── Physics control panel (lets judges tweak layout live) ───────────────
    net.show_buttons(filter_=["physics", "nodes", "edges"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    net.write_html(out_path, open_browser=False, notebook=False)
    print(f"[PyVis] Interactive graph saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else "../output/doc_graph.pkl"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "../output/images/graph_interactive.html"

    graph = DocumentGraph.load(pkl_path)
    print(f"[PyVis] Loaded graph: {graph}")
    build_interactive_graph(graph, out_path)