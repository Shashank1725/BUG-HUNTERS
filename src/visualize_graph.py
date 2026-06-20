"""
visualize_graph.py  ─  Person 2: Graph Visualization
-----------------------------------------------------
Generates visual graph diagrams from the DocumentGraph.

Outputs (all saved to output_dir):
  graph_full.png        — complete graph (all nodes + edges)
  graph_by_type.png     — nodes colored by element type
  graph_per_page.png    — one subplot per page
  graph_edge_types.png  — one subplot per edge type

Usage:
  python visualize_graph.py output/doc_graph.pkl
  python visualize_graph.py output/doc_graph.pkl --output ../output
"""

import sys
import os
import argparse
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from graph import DocumentGraph
from schema import ElementType, EdgeType


# ── Color maps ────────────────────────────────────────────────────────────────

NODE_COLORS = {
    ElementType.HEADING   : "#FF6B6B",   # red
    ElementType.PARAGRAPH : "#4ECDC4",   # teal
    ElementType.TABLE     : "#45B7D1",   # blue
    ElementType.IMAGE     : "#96CEB4",   # green
    ElementType.CAPTION   : "#FFEAA7",   # yellow
    ElementType.LIST_ITEM : "#DDA0DD",   # plum
}

EDGE_COLORS = {
    EdgeType.SEQUENTIAL   : "#888888",   # grey
    EdgeType.PARENT_CHILD : "#FF6B6B",   # red
    EdgeType.SAME_SECTION : "#4ECDC4",   # teal
    EdgeType.CAPTION_OF   : "#FFD700",   # gold
    EdgeType.FOOTNOTE_REF : "#FF8C00",   # orange
    EdgeType.CROSS_REF    : "#9B59B6",   # purple
    EdgeType.CO_REFERENCE : "#3498DB",   # blue
    EdgeType.PROXIMITY    : "#95A5A6",   # light grey
}

DEFAULT_NODE_COLOR = "#CCCCCC"
DEFAULT_EDGE_COLOR = "#AAAAAA"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _node_color(node):
    return NODE_COLORS.get(node.type, DEFAULT_NODE_COLOR)

def _short_label(node):
    content = node.content if isinstance(node.content, str) else str(node.content)
    text = content[:20].replace("\n", " ").strip()
    return f"{node.type.value[0].upper()}\n{text}"

def _build_nx(graph: DocumentGraph):
    """Rebuild a plain networkx graph from DocumentGraph for drawing."""
    G = nx.DiGraph()
    for n in graph.all_nodes():
        G.add_node(n.element_id,
                   label=_short_label(n),
                   color=_node_color(n),
                   node_type=n.type,
                   page=n.page)
    for src, tgt, data in graph._g.edges(data=True):
        et = data.get("edge_type", EdgeType.SEQUENTIAL)
        color = EDGE_COLORS.get(et, DEFAULT_EDGE_COLOR)
        G.add_edge(src, tgt, edge_type=et, color=color, weight=data.get("weight", 1.0))
    return G


def _legend_patches(colors_map, label_fn=lambda x: x.value):
    return [mpatches.Patch(color=c, label=label_fn(k)) for k, c in colors_map.items()]


# ── Plot 1: Full graph ────────────────────────────────────────────────────────

def plot_full_graph(graph: DocumentGraph, out_path: str):
    G = _build_nx(graph)
    if len(G.nodes) == 0:
        print("[Viz] No nodes to draw.")
        return

    fig, ax = plt.subplots(figsize=(20, 14))
    pos = nx.spring_layout(G, k=2.5, seed=42)

    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    edge_colors = [G.edges[e]["color"] for e in G.edges]
    labels      = {n: G.nodes[n]["label"] for n in G.nodes}

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=800, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           arrows=True, arrowsize=15,
                           alpha=0.6, ax=ax,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=6, ax=ax)

    ax.legend(handles=_legend_patches(NODE_COLORS), title="Node Type",
              loc="upper left", fontsize=8)
    ax.set_title("Document Graph — Full View", fontsize=16, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved → {out_path}")


# ── Plot 2: Nodes colored by type, edges colored by type ──────────────────────

def plot_by_type(graph: DocumentGraph, out_path: str):
    G = _build_nx(graph)
    if len(G.nodes) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    pos = nx.spring_layout(G, k=2.5, seed=42)

    # Left: node types
    ax = axes[0]
    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=600, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#CCCCCC",
                           arrows=True, arrowsize=12, alpha=0.4, ax=ax)
    nx.draw_networkx_labels(G, pos,
                            labels={n: G.nodes[n]["label"] for n in G.nodes},
                            font_size=6, ax=ax)
    ax.legend(handles=_legend_patches(NODE_COLORS), title="Node Type",
              loc="upper left", fontsize=8)
    ax.set_title("Nodes by Element Type", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Right: edge types
    ax = axes[1]
    nx.draw_networkx_nodes(G, pos, node_color="#DDDDDD",
                           node_size=600, alpha=0.7, ax=ax)
    edge_colors = [G.edges[e]["color"] for e in G.edges]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           arrows=True, arrowsize=12, alpha=0.7, ax=ax,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos,
                            labels={n: G.nodes[n]["label"] for n in G.nodes},
                            font_size=6, ax=ax)
    ax.legend(handles=_legend_patches(EDGE_COLORS), title="Edge Type",
              loc="upper left", fontsize=8)
    ax.set_title("Edges by Relationship Type", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.suptitle("Document Graph — Type Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved → {out_path}")


# ── Plot 3: Per-page subgraphs ────────────────────────────────────────────────

def plot_per_page(graph: DocumentGraph, out_path: str):
    pages = sorted(set(n.page for n in graph.all_nodes()))
    if not pages:
        return

    cols = min(3, len(pages))
    rows = (len(pages) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(8 * cols, 7 * rows))
    axes = [axes] if len(pages) == 1 else axes.flatten()

    G_full = _build_nx(graph)

    for idx, page in enumerate(pages):
        ax = axes[idx]
        page_nodes = [n.element_id for n in graph.nodes_on_page(page)]
        G_sub = G_full.subgraph(page_nodes)

        if len(G_sub.nodes) == 0:
            ax.axis("off")
            continue

        pos = nx.spring_layout(G_sub, k=2.0, seed=42)
        node_colors = [G_sub.nodes[n]["color"] for n in G_sub.nodes]
        edge_colors = [G_sub.edges[e]["color"] for e in G_sub.edges]

        nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors,
                               node_size=500, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G_sub, pos, edge_color=edge_colors,
                               arrows=True, arrowsize=12, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G_sub, pos,
                                labels={n: G_sub.nodes[n]["label"] for n in G_sub.nodes},
                                font_size=7, ax=ax)
        ax.set_title(f"Page {page}  ({len(G_sub.nodes)} nodes)",
                     fontsize=12, fontweight="bold")
        ax.axis("off")

    # Hide unused subplots
    for idx in range(len(pages), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Document Graph — Per Page View", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved → {out_path}")


# ── Plot 4: Stats bar chart ───────────────────────────────────────────────────

def plot_stats(graph: DocumentGraph, out_path: str):
    stats = graph.stats()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Node types bar
    ax = axes[0]
    nt = stats["node_types"]
    colors = [NODE_COLORS.get(ElementType(k), DEFAULT_NODE_COLOR) for k in nt]
    bars = ax.bar(list(nt.keys()), list(nt.values()), color=colors, edgecolor="white")
    ax.bar_label(bars, fontsize=10, fontweight="bold")
    ax.set_title("Node Count by Type", fontsize=13, fontweight="bold")
    ax.set_xlabel("Element Type")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

    # Edge types bar
    ax = axes[1]
    et = stats["edge_types"]
    colors = [EDGE_COLORS.get(EdgeType(k), DEFAULT_EDGE_COLOR) for k in et]
    bars = ax.bar(list(et.keys()), list(et.values()), color=colors, edgecolor="white")
    ax.bar_label(bars, fontsize=10, fontweight="bold")
    ax.set_title("Edge Count by Type", fontsize=13, fontweight="bold")
    ax.set_xlabel("Edge Type")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Document Graph — Statistics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def visualize(pkl_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    graph = DocumentGraph.load(pkl_path)

    print(f"[Viz] Graph: {graph}")
    print(f"[Viz] Generating visualizations → {output_dir}\n")

    plot_full_graph(graph, os.path.join(output_dir, "graph_full.png"))
    plot_by_type   (graph, os.path.join(output_dir, "graph_by_type.png"))
    plot_per_page  (graph, os.path.join(output_dir, "graph_per_page.png"))
    plot_stats     (graph, os.path.join(output_dir, "graph_stats.png"))

    print(f"\n[Viz] Done! 4 images saved to: {os.path.abspath(output_dir)}")
    print("  graph_full.png     — complete graph")
    print("  graph_by_type.png  — nodes & edges colored by type")
    print("  graph_per_page.png — one subplot per page")
    print("  graph_stats.png    — bar chart statistics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DocumentGraph")
    parser.add_argument("pkl", help="Path to doc_graph.pkl")
    parser.add_argument("--output", default=None,
                        help="Output folder (default: same folder as pkl)")
    args = parser.parse_args()

    out_dir = args.output or os.path.dirname(os.path.abspath(args.pkl))
    visualize(args.pkl, out_dir)