# """
# heuristics.py  ─  Rule-based relationship extraction
# -----------------------------------------------------
# Consumes a list[DocElement] (Person 1's parser output) and returns
# a DocumentGraph with typed edges from six heuristics:

#   1. Sequential ordering     SEQUENTIAL    reading-order adjacency
#   2. Heading → section       PARENT_CHILD  heading owns its children
#                              SAME_SECTION  siblings share the same heading
#   3. Caption–figure linking  CAPTION_OF    nearest image/table on same page
#   4. Proximity               PROXIMITY     spatially close elements
#   5. Cross-reference parsing CROSS_REF     "see Table 3", "Figure 2 shows…"
#   6. Footnote references     FOOTNOTE_REF  [1] / ^1 markers in paragraphs

# Integration note
# ----------------
# Replace local imports with:
#     from schema  import DocElement, ElementType, BoundingBox
#     from p2_schema import DocumentEdge, EdgeType
# """

# from __future__ import annotations

# import re
# from dataclasses import dataclass
# from typing import Optional

# # from schema_p1 import DocElement, ElementType   # ← Person 1's types
# # from p2_schema  import DocumentEdge, EdgeType   # ← Person 2's additions
# from schema import DocElement, ElementType, DocumentEdge, EdgeType  # ← shared repo
# from graph       import DocumentGraph


# # ── Configuration ─────────────────────────────────────────────────────────────

# @dataclass
# class HeuristicConfig:
#     sequential_gap_threshold     : float = 40.0   # max vertical gap (pts) → SEQUENTIAL
#     proximity_distance_threshold : float = 150.0  # max centre-dist (pts)  → PROXIMITY
#     caption_proximity_threshold  : float = 60.0   # max y-gap (pts)        → CAPTION_OF
#     heuristic_weight             : float = 0.8    # default edge confidence
#     proximity_weight             : float = 0.5    # base weight for proximity edges


# # ── Builder ───────────────────────────────────────────────────────────────────

# class HeuristicGraphBuilder:
#     """
#     Build a DocumentGraph from a flat list[DocElement].

#     Usage (within the integrated project)
#     --------------------------------------
#     from heuristics import HeuristicGraphBuilder
#     graph = HeuristicGraphBuilder().build(parsed_doc.elements)
#     """

#     # Patterns
#     _CROSS_REF_RE = re.compile(
#         r'\b(?:see|refer to|shown in|as in|cf\.?)\s+'
#         r'(?P<label>(?:Table|Figure|Fig\.?|Section|Appendix)\s*\d+)',
#         re.IGNORECASE,
#     )
#     _FOOTNOTE_RE = re.compile(r'\[(\d+)\]|\^(\d+)')

#     def __init__(self, config: Optional[HeuristicConfig] = None) -> None:
#         self.cfg = config or HeuristicConfig()

#     # ── Public entry ──────────────────────────────────────────────────────────

#     def build(self, elements: list[DocElement]) -> DocumentGraph:
#         """
#         Main entry point.  Pass parsed_doc.elements from Person 1.
#         Returns a fully populated DocumentGraph.
#         """
#         g = DocumentGraph()
#         g.add_nodes(elements)

#         ordered = self._reading_order(elements)

#         edges: list[DocumentEdge] = []
#         edges += self._sequential_edges(ordered)
#         edges += self._heading_section_edges(ordered)
#         edges += self._caption_figure_edges(ordered)
#         edges += self._proximity_edges(ordered)
#         edges += self._cross_reference_edges(ordered)
#         edges += self._footnote_edges(ordered)

#         g.add_edges(edges)
#         print(f"[HeuristicBuilder] {g}")
#         return g

#     # ── Sorting ───────────────────────────────────────────────────────────────

#     @staticmethod
#     def _reading_order(elements: list[DocElement]) -> list[DocElement]:
#         """Sort by page → y0 → x0.  Elements without a BoundingBox sort last."""
#         def key(e: DocElement):
#             bb = e.position
#             return (e.page, bb.y0 if bb else 1e9, bb.x0 if bb else 1e9)
#         return sorted(elements, key=key)

#     # ── 1. Sequential edges ───────────────────────────────────────────────────

#     def _sequential_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
#         edges = []
#         for i in range(len(els) - 1):
#             a, b = els[i], els[i + 1]
#             # Same page: skip if vertical gap is too large (column break, etc.)
#             if a.page == b.page and a.position and b.position:
#                 gap = b.position.y0 - a.position.y1
#                 if gap > self.cfg.sequential_gap_threshold:
#                     continue
#             edges.append(DocumentEdge(
#                 source_id=a.element_id,
#                 target_id=b.element_id,
#                 edge_type=EdgeType.SEQUENTIAL,
#                 weight=self.cfg.heuristic_weight,
#             ))
#         return edges

#     # ── 2. Heading → section ──────────────────────────────────────────────────

#     def _heading_section_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
#         """
#         For each heading walk forward collecting its section members (until a
#         heading of equal or higher level appears).  Emit:
#           PARENT_CHILD  heading → every member
#           SAME_SECTION  between every pair of members
#         """
#         edges = []
#         for h_idx, heading in enumerate(els):
#             if heading.type != ElementType.HEADING:
#                 continue
#             level = heading.heading_level or 1
#             members: list[DocElement] = []

#             for el in els[h_idx + 1:]:
#                 if el.type == ElementType.HEADING and (el.heading_level or 1) <= level:
#                     break
#                 members.append(el)

#             for m in members:
#                 edges.append(DocumentEdge(
#                     source_id=heading.element_id,
#                     target_id=m.element_id,
#                     edge_type=EdgeType.PARENT_CHILD,
#                     weight=self.cfg.heuristic_weight,
#                 ))

#             for i, ma in enumerate(members):
#                 for mb in members[i + 1:]:
#                     edges.append(DocumentEdge(
#                         source_id=ma.element_id,
#                         target_id=mb.element_id,
#                         edge_type=EdgeType.SAME_SECTION,
#                         weight=round(self.cfg.heuristic_weight * 0.7, 3),
#                     ))
#         return edges

#     # ── 3. Caption ↔ figure / table ───────────────────────────────────────────

#     def _caption_figure_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
#         """
#         Link each CAPTION to the spatially nearest IMAGE or TABLE on the same
#         page within caption_proximity_threshold vertical points.
#         """
#         edges = []
#         captions = [e for e in els if e.type == ElementType.CAPTION]
#         figures  = [e for e in els if e.type in (ElementType.IMAGE, ElementType.TABLE)]

#         for cap in captions:
#             if not cap.position:
#                 continue
#             best: Optional[DocElement] = None
#             best_dist = float("inf")

#             for fig in figures:
#                 if fig.page != cap.page or not fig.position:
#                     continue
#                 dist = abs(self._cy(cap) - self._cy(fig))
#                 if dist < best_dist and dist < self.cfg.caption_proximity_threshold:
#                     best_dist, best = dist, fig

#             if best:
#                 edges.append(DocumentEdge(
#                     source_id=cap.element_id,
#                     target_id=best.element_id,
#                     edge_type=EdgeType.CAPTION_OF,
#                     weight=1.0,
#                     metadata={"distance_pts": round(best_dist, 2)},
#                 ))
#         return edges

#     # ── 4. Proximity edges ────────────────────────────────────────────────────

#     def _proximity_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
#         """
#         Add PROXIMITY edges between elements that are spatially close
#         on the same page.  Weight decreases linearly with distance.
#         """
#         edges = []
#         by_page: dict[int, list[DocElement]] = {}
#         for e in els:
#             by_page.setdefault(e.page, []).append(e)

#         for page_els in by_page.values():
#             positioned = [e for e in page_els if e.position]
#             for i, a in enumerate(positioned):
#                 for b in positioned[i + 1:]:
#                     dist = self._bbox_dist(a, b)
#                     if dist < self.cfg.proximity_distance_threshold:
#                         w = self.cfg.proximity_weight * (
#                             1 - dist / self.cfg.proximity_distance_threshold
#                         )
#                         edges.append(DocumentEdge(
#                             source_id=a.element_id,
#                             target_id=b.element_id,
#                             edge_type=EdgeType.PROXIMITY,
#                             weight=round(w, 3),
#                             metadata={"distance_pts": round(dist, 2)},
#                         ))
#         return edges

#     # ── 5. Cross-reference edges ──────────────────────────────────────────────

#     # def _cross_reference_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
#     #     """
#     #     Parse paragraph text for patterns like "see Table 3" or "Figure 2 shows"
#     #     and emit CROSS_REF edges to the matching caption/heading element.
#     #     """
#     #     edges = []

#     #     # Build label → element_id map from captions and headings
#     #     label_map: dict[str, str] = {}
#     #     for e in els:
#     #         m = re.match(
#     #             r'^(Table|Figure|Fig\.?|Section|Appendix)\s*(\d+)',
#     #             e.content.strip(), re.IGNORECASE,
#     #         )
#     #         if m:
#     #             key1 = f"{m.group(1).capitalize()} {m.group(2)}"
#     #             key2 = f"Fig. {m.group(2)}"   # "Fig. N" alias
#     #             label_map[key1] = e.element_id
#     #             label_map[key2] = e.element_id

#     #     text_types = {ElementType.PARAGRAPH, ElementType.LIST_ITEM}
#     #     for el in els:
#     #         if el.type not in text_types:
#     #             continue
#     #         for m in self._CROSS_REF_RE.finditer(el.content):
#     #             raw   = m.group("label")
#     #             label = re.sub(r'\bFig\.?\b', 'Figure', raw, flags=re.IGNORECASE).strip()
#     #             tgt   = label_map.get(label)
#     #             if tgt and tgt != el.element_id:
#     #                 edges.append(DocumentEdge(
#     #                     source_id=el.element_id,
#     #                     target_id=tgt,
#     #                     edge_type=EdgeType.CROSS_REF,
#     #                     weight=self.cfg.heuristic_weight,
#     #                     metadata={"ref_text": m.group(0)},
#     #                 ))
#     #     return edges

# def _cross_reference_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
#     edges = []
#     label_map: dict[str, str] = {}
#     for e in els:
#         # Safety check — skip if content is not a plain string
#         if not isinstance(self._safe_content(e), str):
#             continue
#         m = re.match(
#             r'^(Table|Figure|Fig\.?|Section|Appendix)\s*(\d+)',
#             self._safe_content(e).strip(), re.IGNORECASE,
#         )
#         if m:
#             key1 = f"{m.group(1).capitalize()} {m.group(2)}"
#             key2 = f"Fig. {m.group(2)}"
#             label_map[key1] = e.element_id
#             label_map[key2] = e.element_id

#     text_types = {ElementType.PARAGRAPH, ElementType.LIST_ITEM}
#     for el in els:
#         if el.type not in text_types:
#             continue
#         # Safety check — skip if content is not a plain string
#         if not isinstance(self._safe_content(el), str):
#             continue
#         for m in self._CROSS_REF_RE.finditer(self._safe_content(el)):
#             raw   = m.group("label")
#             label = re.sub(r'\bFig\.?\b', 'Figure', raw, flags=re.IGNORECASE).strip()
#             tgt   = label_map.get(label)
#             if tgt and tgt != el.element_id:
#                 edges.append(DocumentEdge(
#                     source_id=el.element_id,
#                     target_id=tgt,
#                     edge_type=EdgeType.CROSS_REF,
#                     weight=self.cfg.heuristic_weight,
#                     metadata={"ref_text": m.group(0)},
#                 ))
#     return edges

#     # ── 6. Footnote reference edges ───────────────────────────────────────────

#     def _footnote_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
#         """
#         Detect footnote markers like [1] or ^2 in paragraph text and link them
#         to footnote elements.

#         Person 1 marks footnotes with metadata["is_footnote"] = True.
#         The marker number is extracted from the footnote's content.
#         """
#         edges = []
#         footnote_map: dict[str, str] = {}   # marker_number → element_id
#         for e in els:
#             if e.metadata.get("is_footnote"):
#                 m = re.search(r'\d+', self._safe_content(e))
#                 if m:
#                     footnote_map[m.group()] = e.element_id

#         for el in els:
#             if el.type != ElementType.PARAGRAPH:
#                 continue
#             for m in self._FOOTNOTE_RE.finditer(self._safe_content(e)):
#                 marker = next(g for g in m.groups() if g is not None)
#                 if marker in footnote_map:
#                     edges.append(DocumentEdge(
#                         source_id=el.element_id,
#                         target_id=footnote_map[marker],
#                         edge_type=EdgeType.FOOTNOTE_REF,
#                         weight=0.9,
#                         metadata={"marker": marker},
#                     ))
#         return edges

#     # ── Geometry helpers ──────────────────────────────────────────────────────

#     @staticmethod
#     def _cy(e: DocElement) -> float:
#         bb = e.position
#         return (bb.y0 + bb.y1) / 2 if bb else 0.0

#     @staticmethod
#     def _bbox_dist(a: DocElement, b: DocElement) -> float:
#         def cx(e): return (e.position.x0 + e.position.x1) / 2
#         def cy(e): return (e.position.y0 + e.position.y1) / 2
#         return ((cx(a) - cx(b)) ** 2 + (cy(a) - cy(b)) ** 2) ** 0.5

#     @staticmethod
#     def _safe_content(e: DocElement) -> str:
#         """Always return a plain string regardless of what the parser stored."""
#         if isinstance(e.content, str):
#             return e.content
#         if isinstance(e.content, (list, tuple)):
#             return " ".join(str(x) for x in e.content)
#         return str(e.content) if e.content is not None else ""

"""
heuristics.py  ─  Rule-based relationship extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from schema import DocElement, ElementType, DocumentEdge, EdgeType
from graph  import DocumentGraph


@dataclass
class HeuristicConfig:
    sequential_gap_threshold     : float = 40.0
    proximity_distance_threshold : float = 150.0
    caption_proximity_threshold  : float = 60.0
    heuristic_weight             : float = 0.8
    proximity_weight             : float = 0.5


class HeuristicGraphBuilder:

    _CROSS_REF_RE = re.compile(
        r'\b(?:see|refer to|shown in|as in|cf\.?)\s+'
        r'(?P<label>(?:Table|Figure|Fig\.?|Section|Appendix)\s*\d+)',
        re.IGNORECASE,
    )
    _FOOTNOTE_RE = re.compile(r'\[(\d+)\]|\^(\d+)')

    def __init__(self, config: Optional[HeuristicConfig] = None) -> None:
        self.cfg = config or HeuristicConfig()

    # ── Content safety helper ─────────────────────────────────────────────────

    @staticmethod
    def _safe_content(e: DocElement) -> str:
        """
        Always return a plain string from e.content.
        Guards against parsers that store tuples, lists, or None.
        """
        if isinstance(e.content, str):
            return e.content
        if isinstance(e.content, (list, tuple)):
            return " ".join(str(x) for x in e.content)
        return str(e.content) if e.content is not None else ""

    # ── Public entry ──────────────────────────────────────────────────────────

    def build(self, elements: list[DocElement]) -> DocumentGraph:
        g = DocumentGraph()
        g.add_nodes(elements)

        ordered = self._reading_order(elements)

        edges: list[DocumentEdge] = []
        edges += self._sequential_edges(ordered)
        edges += self._heading_section_edges(ordered)
        edges += self._caption_figure_edges(ordered)
        edges += self._proximity_edges(ordered)
        edges += self._cross_reference_edges(ordered)
        edges += self._footnote_edges(ordered)

        g.add_edges(edges)
        print(f"[HeuristicBuilder] {g}")
        return g

    # ── Sorting ───────────────────────────────────────────────────────────────

    @staticmethod
    def _reading_order(elements: list[DocElement]) -> list[DocElement]:
        def key(e: DocElement):
            bb = e.position
            return (e.page, bb.y0 if bb else 1e9, bb.x0 if bb else 1e9)
        return sorted(elements, key=key)

    # ── 1. Sequential ─────────────────────────────────────────────────────────

    def _sequential_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
        edges = []
        for i in range(len(els) - 1):
            a, b = els[i], els[i + 1]
            if a.page == b.page and a.position and b.position:
                gap = b.position.y0 - a.position.y1
                if gap > self.cfg.sequential_gap_threshold:
                    continue
            edges.append(DocumentEdge(
                source_id=a.element_id,
                target_id=b.element_id,
                edge_type=EdgeType.SEQUENTIAL,
                weight=self.cfg.heuristic_weight,
            ))
        return edges

    # ── 2. Heading → section ──────────────────────────────────────────────────

    def _heading_section_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
        # No content access here — no _safe_content needed
        edges = []
        for h_idx, heading in enumerate(els):
            if heading.type != ElementType.HEADING:
                continue
            level = heading.heading_level or 1
            members: list[DocElement] = []

            for el in els[h_idx + 1:]:
                if el.type == ElementType.HEADING and (el.heading_level or 1) <= level:
                    break
                members.append(el)

            for m in members:
                edges.append(DocumentEdge(
                    source_id=heading.element_id,
                    target_id=m.element_id,
                    edge_type=EdgeType.PARENT_CHILD,
                    weight=self.cfg.heuristic_weight,
                ))

            # for i, ma in enumerate(members):
            #     for mb in members[i + 1:]:
            #         edges.append(DocumentEdge(
            #             source_id=ma.element_id,
            #             target_id=mb.element_id,
            #             edge_type=EdgeType.SAME_SECTION,
            #             weight=round(self.cfg.heuristic_weight * 0.7, 3),
            #         ))
            SAME_SECTION_WINDOW = 3
            for i, ma in enumerate(members):
                for mb in members[i + 1:i + 1 + SAME_SECTION_WINDOW]:
                    edges.append(DocumentEdge(
                        source_id=ma.element_id,
                        target_id=mb.element_id,
                        edge_type=EdgeType.SAME_SECTION,
                        weight=round(self.cfg.heuristic_weight * 0.7, 3),
                    ))
        return edges

    # ── 3. Caption ↔ figure / table ───────────────────────────────────────────

    def _caption_figure_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
        # No content access here — no _safe_content needed
        edges = []
        captions = [e for e in els if e.type == ElementType.CAPTION]
        figures  = [e for e in els if e.type in (ElementType.IMAGE, ElementType.TABLE)]

        for cap in captions:
            if not cap.position:
                continue
            best: Optional[DocElement] = None
            best_dist = float("inf")

            for fig in figures:
                if fig.page != cap.page or not fig.position:
                    continue
                dist = abs(self._cy(cap) - self._cy(fig))
                if dist < best_dist and dist < self.cfg.caption_proximity_threshold:
                    best_dist, best = dist, fig

            if best:
                edges.append(DocumentEdge(
                    source_id=cap.element_id,
                    target_id=best.element_id,
                    edge_type=EdgeType.CAPTION_OF,
                    weight=1.0,
                    metadata={"distance_pts": round(best_dist, 2)},
                ))
        return edges

    # ── 4. Proximity ──────────────────────────────────────────────────────────

    def _proximity_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
        # No content access here — no _safe_content needed
        edges = []
        by_page: dict[int, list[DocElement]] = {}
        for e in els:
            by_page.setdefault(e.page, []).append(e)

        for page_els in by_page.values():
            positioned = [e for e in page_els if e.position]
            for i, a in enumerate(positioned):
                for b in positioned[i + 1:]:
                    dist = self._bbox_dist(a, b)
                    if dist < self.cfg.proximity_distance_threshold:
                        w = self.cfg.proximity_weight * (
                            1 - dist / self.cfg.proximity_distance_threshold
                        )
                        edges.append(DocumentEdge(
                            source_id=a.element_id,
                            target_id=b.element_id,
                            edge_type=EdgeType.PROXIMITY,
                            weight=round(w, 3),
                            metadata={"distance_pts": round(dist, 2)},
                        ))
        return edges

    # ── 5. Cross-references ───────────────────────────────────────────────────

    def _cross_reference_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
        edges = []
        label_map: dict[str, str] = {}

        for e in els:
            content = self._safe_content(e)   # ← safe
            m = re.match(
                r'^(Table|Figure|Fig\.?|Section|Appendix)\s*(\d+)',
                content.strip(), re.IGNORECASE,
            )
            if m:
                key1 = f"{m.group(1).capitalize()} {m.group(2)}"
                key2 = f"Fig. {m.group(2)}"
                label_map[key1] = e.element_id
                label_map[key2] = e.element_id

        text_types = {ElementType.PARAGRAPH, ElementType.LIST_ITEM}
        for el in els:
            if el.type not in text_types:
                continue
            content = self._safe_content(el)   # ← safe
            for m in self._CROSS_REF_RE.finditer(content):
                raw   = m.group("label")
                label = re.sub(r'\bFig\.?\b', 'Figure', raw, flags=re.IGNORECASE).strip()
                tgt   = label_map.get(label)
                if tgt and tgt != el.element_id:
                    edges.append(DocumentEdge(
                        source_id=el.element_id,
                        target_id=tgt,
                        edge_type=EdgeType.CROSS_REF,
                        weight=self.cfg.heuristic_weight,
                        metadata={"ref_text": m.group(0)},
                    ))
        return edges

    # ── 6. Footnote references ────────────────────────────────────────────────

    def _footnote_edges(self, els: list[DocElement]) -> list[DocumentEdge]:
        edges = []
        footnote_map: dict[str, str] = {}

        for e in els:
            if e.metadata.get("is_footnote"):
                content = self._safe_content(e)   # ← safe
                m = re.search(r'\d+', content)
                if m:
                    footnote_map[m.group()] = e.element_id

        for el in els:
            if el.type != ElementType.PARAGRAPH:
                continue
            content = self._safe_content(el)   # ← safe
            for m in self._FOOTNOTE_RE.finditer(content):
                marker = next(g for g in m.groups() if g is not None)
                if marker in footnote_map:
                    edges.append(DocumentEdge(
                        source_id=el.element_id,
                        target_id=footnote_map[marker],
                        edge_type=EdgeType.FOOTNOTE_REF,
                        weight=0.9,
                        metadata={"marker": marker},
                    ))
        return edges

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _cy(e: DocElement) -> float:
        bb = e.position
        return (bb.y0 + bb.y1) / 2 if bb else 0.0

    @staticmethod
    def _bbox_dist(a: DocElement, b: DocElement) -> float:
        def cx(e): return (e.position.x0 + e.position.x1) / 2
        def cy(e): return (e.position.y0 + e.position.y1) / 2
        return ((cx(a) - cx(b)) ** 2 + (cy(a) - cy(b)) ** 2) ** 0.5