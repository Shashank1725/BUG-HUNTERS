"""
document_graph.py — Person 2: Relationship Modeling & Graph Construction

Defines the schema for the document element graph and a DocumentGraph class
that wraps networkx to build, query, and export it.

Design notes:
- Nodes are the elements Person 1 already produces (paragraph/table/image/
  caption/heading) — we don't redefine element shape, we just index it.
- We use a MultiDiGraph: directed because relations like "explains" have a
  clear source -> target meaning, and Multi- because two elements can be
  connected by more than one relation type at once (rare, but schema should
  allow it rather than silently dropping an edge).
- Every edge carries its own confidence + method, separate from the node's
  own parse/caption confidence. A heuristic edge and a node's caption
  confidence are different things and Person 4 may want both when deciding
  how much to trust an answer.
"""

from __future__ import annotations

import json
import os
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

import networkx as nx


# ---------------------------------------------------------------------------
# Relation vocabulary
# ---------------------------------------------------------------------------

class Relation(str, Enum):
    """Edge types. Values match the problem statement's own examples."""
    CONTAINS = "contains"            # heading -> paragraph/table/image under it
    EXPLAINS = "explains"            # paragraph -> table/chart it describes
    BELONGS_TO = "belongs_to"        # caption -> image/table it captions
    VISUALIZED_BY = "visualized_by"  # table -> chart that visualizes it
    REFERENCES = "references"        # paragraph -> element via "see Table 2", footnote, etc.
    FOLLOWS = "follows"              # weak fallback: same section, document order
    RELATED_TO = "related_to"        # generic weak link when nothing more specific applies


class EdgeMethod(str, Enum):
    """How an edge was discovered — lets Person 3/4 weight trust appropriately."""
    PARSER = "parser"        # came straight from Person 1's parent_id field
    HEURISTIC = "heuristic"  # proximity / regex / shared-header rule
    ML = "ml"                # co-reference or learned relationship model


# ---------------------------------------------------------------------------
# Edge record
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    source: str
    target: str
    relation: Relation
    confidence: float = 1.0
    method: EdgeMethod = EdgeMethod.HEURISTIC
    evidence: Optional[str] = None  # e.g. matched text, or "same page + adjacent bbox"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation.value,
            "confidence": self.confidence,
            "method": self.method.value,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# DocumentGraph
# ---------------------------------------------------------------------------

class DocumentGraph:
    """
    Wraps a networkx.MultiDiGraph of document elements.

    Nodes store the full element dict from Person 1 (element_id, type,
    content, page, position, heading_level, parent_id, confidence, metadata).
    We intentionally do NOT store the embedding vector on the graph node to
    keep graph exports small — Person 3 owns the vector index and can look
    embeddings up by element_id if needed.
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.g = nx.MultiDiGraph()

    # -- building -----------------------------------------------------------

    def add_node(self, element: dict) -> None:
        """Add one element from Person 1's parsed output as a graph node."""
        node_id = element["element_id"]
        # Keep everything except the embedding (too large for graph export).
        attrs = {k: v for k, v in element.items() if k != "embedding"}
        self.g.add_node(node_id, **attrs)

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self.g or edge.target not in self.g:
            raise ValueError(
                f"Cannot add edge {edge.source} -> {edge.target}: "
                f"one or both nodes not in graph"
            )
        self.g.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation.value,
            confidence=edge.confidence,
            method=edge.method.value,
            evidence=edge.evidence,
        )

    def load_from_elements(self, elements: list[dict]) -> None:
        """
        Load a full element list straight from Person 1's parsed JSON
        ("elements" array). Adds every node, then derives CONTAINS edges
        for free from the parent_id field Person 1 already populates —
        zero heuristic work needed for that relation.
        """
        for element in elements:
            self.add_node(element)

        for element in elements:
            parent_id = element.get("parent_id")
            if parent_id:
                self.add_edge(Edge(
                    source=parent_id,
                    target=element["element_id"],
                    relation=Relation.CONTAINS,
                    confidence=element.get("confidence", 1.0),
                    method=EdgeMethod.PARSER,
                    evidence="parent_id from parser",
                ))

    @classmethod
    def from_parsed_json(cls, path: str) -> "DocumentGraph":
        """Build a graph directly from one of Person 1's *_parsed.json files."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        doc_id = data.get("source_file", path)
        graph = cls(doc_id=doc_id)
        graph.load_from_elements(data["elements"])
        return graph

    # -- querying (what Person 3 will actually call) -------------------------

    def get_node(self, element_id: str) -> dict:
        return dict(self.g.nodes[element_id])

    def neighbors(
        self,
        element_id: str,
        relation: Optional[Relation] = None,
        direction: str = "both",  # "out", "in", or "both"
        hops: int = 1,
    ) -> list[dict]:
        """
        Return neighboring elements within `hops` steps, optionally filtered
        to a single relation type. This is the main entry point Person 3
        traverses when assembling a multi-element context bundle for a query.
        """
        frontier = {element_id}
        visited = {element_id}
        results: list[str] = []

        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                edge_iters = []
                if direction in ("out", "both"):
                    edge_iters.append(self.g.out_edges(node, data=True))
                if direction in ("in", "both"):
                    edge_iters.append(self.g.in_edges(node, data=True))

                for edges in edge_iters:
                    for u, v, data in edges:
                        other = v if u == node else u
                        if relation and data.get("relation") != relation.value:
                            continue
                        if other not in visited:
                            visited.add(other)
                            next_frontier.add(other)
                            results.append(other)
            frontier = next_frontier

        return [self.get_node(n) for n in results]

    def edges_for(self, element_id: str) -> list[dict]:
        """All edges (in + out) touching a node, with relation metadata."""
        out = [
            {"source": u, "target": v, **data}
            for u, v, data in self.g.out_edges(element_id, data=True)
        ]
        inn = [
            {"source": u, "target": v, **data}
            for u, v, data in self.g.in_edges(element_id, data=True)
        ]
        return out + inn

    # -- export / import (the contract with Person 3) ------------------------

    def to_dict(self) -> dict:
        nodes = [
            {"element_id": n, **{k: v for k, v in d.items()}}
            for n, d in self.g.nodes(data=True)
        ]
        edges = [
            {"source": u, "target": v, **d}
            for u, v, d in self.g.edges(data=True)
        ]
        return {"doc_id": self.doc_id, "nodes": nodes, "edges": edges}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentGraph":
        graph = cls(doc_id=data["doc_id"])
        for node in data["nodes"]:
            graph.g.add_node(node["element_id"], **node)
        for edge in data["edges"]:
            graph.g.add_edge(
                edge["source"], edge["target"],
                relation=edge["relation"], confidence=edge["confidence"],
                method=edge["method"], evidence=edge.get("evidence"),
            )
        return graph

    def stats(self) -> dict:
        relation_counts: dict[str, int] = {}
        for _, _, data in self.g.edges(data=True):
            rel = data["relation"]
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
        return {
            "doc_id": self.doc_id,
            "num_nodes": self.g.number_of_nodes(),
            "num_edges": self.g.number_of_edges(),
            "edges_by_relation": relation_counts,
        }


# ---------------------------------------------------------------------------
# Helper: convert a DocElement (dataclass or Pydantic) to a plain dict
# ---------------------------------------------------------------------------

def _element_to_dict(el) -> dict:
    """
    Safely convert a DocElement object from Person 1's parser to a plain dict.
    Handles both Pydantic models (.model_dump / .dict) and Python dataclasses.
    Falls back to __dict__ for anything else.
    """
    if hasattr(el, "model_dump"):        # Pydantic v2
        d = el.model_dump()
    elif hasattr(el, "dict"):            # Pydantic v1
        d = el.dict()
    elif dataclasses.is_dataclass(el):   # standard dataclass
        d = dataclasses.asdict(el)
    else:
        d = dict(el.__dict__)

    # Enum values (e.g. ElementType) must be plain strings for JSON / graph attrs
    for key, val in d.items():
        if isinstance(val, Enum):
            d[key] = val.value

    return d


# ---------------------------------------------------------------------------
# Entry point — accepts a real .docx file via CLI, no hardcoded data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from docx_parser import parse_docx

    if len(sys.argv) < 2:
        print("Usage: python document_graph.py <path_to_docx>")
        sys.exit(1)

    docx_path = sys.argv[1]

    if not os.path.exists(docx_path):
        print(f"[ERROR] File not found: {docx_path}")
        sys.exit(1)

    # Step 1: Parse the real DOCX using Person 1's parser
    print(f"[*] Parsing: {docx_path}")
    parsed = parse_docx(docx_path)

    # Step 2: Convert DocElement objects → plain dicts
    elements = [_element_to_dict(el) for el in parsed.elements]
    print(f"[✓] Got {len(elements)} elements from parser")

    # Step 3: Build the graph from real elements
    graph = DocumentGraph(doc_id=parsed.source_file)
    graph.load_from_elements(elements)

    # Step 4: Print stats
    print("\n=== Graph Stats ===")
    print(json.dumps(graph.stats(), indent=2))

    # Step 5: Save graph output
    # Use absolute path based on script location to avoid Windows path mixing issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # out_dir = os.path.join(script_dir, "output")
    out_dir = os.path.join(script_dir, "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "document_graph_output.json")
    graph.save(out_path)
    print(f"\n[✓] Saved graph to {out_path}")

    # Step 6: Round-trip verify
    with open(out_path) as f:
        reloaded = DocumentGraph.from_dict(json.load(f))
    assert reloaded.stats()["num_edges"] == graph.stats()["num_edges"], \
        "Round-trip edge count mismatch!"
    print("[✓] Round-trip save/load verified OK.")