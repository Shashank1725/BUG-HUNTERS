"""
graph.py  ─  DocumentGraph
---------------------------
Core queryable graph that Person 3 (Retrieval) will call.

Nodes  →  DocElement objects (produced by Person 1's parsers)
Edges  →  DocumentEdge objects (produced by Person 2's heuristics / ML layer)

Integration note
----------------
Replace the two local imports below with:
    from schema import DocElement, ElementType
    from p2_schema import DocumentEdge, EdgeType
once all modules share a single working directory.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx

# from schema_p1 import DocElement, ElementType   # ← Person 1's types
# from p2_schema  import DocumentEdge, EdgeType   # ← Person 2's additions

from schema import DocElement, ElementType, DocumentEdge, EdgeType  # ← shared repo


class DocumentGraph:
    """
    Directed multi-graph of document elements.

    Public API used by Person 3
    ---------------------------
    graph.neighbors(element_id, edge_types, direction)
    graph.section_context(element_id)
    graph.caption_for(element_id)
    graph.subgraph_around(element_id, hops)
    graph.nodes_by_type(ElementType.TABLE)
    graph.nodes_on_page(3)
    graph.all_nodes()
    graph.get_node(element_id)
    graph.stats()
    graph.save(path) / DocumentGraph.load(path)
    graph.export_json(path)
    """

    def __init__(self) -> None:
        self._g     : nx.MultiDiGraph          = nx.MultiDiGraph()
        self._nodes : dict[str, DocElement]    = {}

    # ── Mutation ──────────────────────────────────────────────────────────────

    # def add_node(self, element: DocElement) -> None:
    #     self._nodes[element.element_id] = element
    #     self._g.add_node(element.element_id,
    #                      element_type=element.type,
    #                      page=element.page)
    def add_node(self, element: DocElement) -> None:
     if element.element_id in self._nodes:
        existing = self._nodes[element.element_id]
        raise ValueError(
            f"Duplicate element_id detected: '{element.element_id}'\n"
            f"  Existing -> type={existing.type}, page={existing.page}, "
            f"content={self._safe_preview(existing)}\n"
            f"  New      -> type={element.type}, page={element.page}, "
            f"content={self._safe_preview(element)}"
        )
     self._nodes[element.element_id] = element
     self._g.add_node(element.element_id,
                     element_type=element.type,
                     page=element.page)

    @staticmethod
    def _safe_preview(e: DocElement) -> str:
        c = e.content
        if isinstance(c, str):
            return c[:60]
        if isinstance(c, (list, tuple)):
            return " ".join(str(x) for x in c)[:60]
        return str(c)[:60] if c is not None else ""

    def add_edge(self, edge: DocumentEdge) -> None:
        self._g.add_edge(edge.source_id, edge.target_id,
                         key=edge.edge_type.value,
                         edge_type=edge.edge_type,
                         weight=edge.weight,
                         **edge.metadata)

    def add_nodes(self, elements: Iterable[DocElement]) -> None:
        for e in elements:
            self.add_node(e)

    def add_edges(self, edges: Iterable[DocumentEdge]) -> None:
        for e in edges:
            self.add_edge(e)

    # ── Basic lookups ─────────────────────────────────────────────────────────

    def get_node(self, element_id: str) -> Optional[DocElement]:
        return self._nodes.get(element_id)

    def all_nodes(self) -> list[DocElement]:
        return list(self._nodes.values())

    def nodes_by_type(self, element_type: ElementType) -> list[DocElement]:
        return [n for n in self._nodes.values() if n.type == element_type]

    def nodes_on_page(self, page: int) -> list[DocElement]:
        return [n for n in self._nodes.values() if n.page == page]

    # ── Traversal API for Person 3 ────────────────────────────────────────────

    def neighbors(
        self,
        element_id : str,
        edge_types : Optional[list[EdgeType]] = None,
        direction  : str = "both",   # "out" | "in" | "both"
    ) -> list[DocElement]:
        """Neighbouring elements, optionally filtered by edge type."""
        result: set[str] = set()
        if direction in ("out", "both"):
            for _, tgt, data in self._g.out_edges(element_id, data=True):
                if edge_types is None or data.get("edge_type") in edge_types:
                    result.add(tgt)
        if direction in ("in", "both"):
            for src, _, data in self._g.in_edges(element_id, data=True):
                if edge_types is None or data.get("edge_type") in edge_types:
                    result.add(src)
        return [self._nodes[nid] for nid in result if nid in self._nodes]

    def section_context(self, element_id: str) -> list[DocElement]:
        """
        All elements that share the same heading ancestor as element_id,
        plus the heading itself.  Useful for assembling section-level context.
        """
        siblings = self.neighbors(element_id, [EdgeType.SAME_SECTION])
        heading  = self.neighbors(element_id, [EdgeType.PARENT_CHILD], direction="in")
        seen: dict[str, DocElement] = {}
        for n in siblings + heading:
            seen[n.element_id] = n
        return list(seen.values())

    def caption_for(self, image_or_table_id: str) -> Optional[DocElement]:
        """Return the caption element linked to an image or table, if any."""
        caps = self.neighbors(image_or_table_id, [EdgeType.CAPTION_OF], direction="in")
        return caps[0] if caps else None

    def subgraph_around(self, element_id: str, hops: int = 2) -> "DocumentGraph":
        """
        Return a new DocumentGraph containing all elements reachable within
        `hops` edges from element_id (undirected BFS).

        Person 3 calls this to build a multi-element context bundle for a query.
        """
        reachable: set[str] = {element_id}
        frontier  = {element_id}
        undirected = self._g.to_undirected()
        for _ in range(hops):
            nxt: set[str] = set()
            for nid in frontier:
                nxt.update(undirected.neighbors(nid))
            frontier   = nxt - reachable
            reachable |= frontier

        sub = DocumentGraph()
        for nid in reachable:
            if nid in self._nodes:
                sub.add_node(self._nodes[nid])
        for src, tgt, data in self._g.edges(data=True):
            if src in reachable and tgt in reachable:
                sub._g.add_edge(src, tgt, **data)
        return sub

    # ── Stats / debug ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        nc: dict[str, int] = {}
        for n in self._nodes.values():
            nc[n.type.value] = nc.get(n.type.value, 0) + 1
        ec: dict[str, int] = {}
        for _, _, d in self._g.edges(data=True):
            et = d.get("edge_type")
            k  = et.value if hasattr(et, "value") else str(et)
            ec[k] = ec.get(k, 0) + 1
        return {
            "total_nodes" : self._g.number_of_nodes(),
            "total_edges" : self._g.number_of_edges(),
            "node_types"  : nc,
            "edge_types"  : ec,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return f"<DocumentGraph nodes={s['total_nodes']} edges={s['total_edges']}>"

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)
        print(f"[DocumentGraph] saved → {p}")

    @staticmethod
    def load(path: str | Path) -> "DocumentGraph":
        with open(path, "rb") as f:
            g = pickle.load(f)
        print(f"[DocumentGraph] loaded ← {path}")
        return g

    def export_json(self, path: str | Path) -> None:
        """Human-readable JSON snapshot — useful for debugging and demos."""
        data = {
            "nodes": [
                {
                    "element_id"   : n.element_id,
                    "type"         : n.type.value,
                    "page"         : n.page,
                    "content"      : n.content[:200],
                    "heading_level": n.heading_level,
                    "parent_id"    : n.parent_id,
                    "metadata"     : n.metadata,
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "source"   : src,
                    "target"   : tgt,
                    "edge_type": d.get("edge_type").value
                                 if hasattr(d.get("edge_type"), "value")
                                 else str(d.get("edge_type")),
                    "weight"   : d.get("weight", 1.0),
                }
                for src, tgt, d in self._g.edges(data=True)
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"[DocumentGraph] JSON exported → {path}")