"""
p2_pipeline.py  ─  Person 2's public entry point
--------------------------------------------------
Takes a ParsedDocument (or a raw list[DocElement]) from Person 1 and
produces a queryable DocumentGraph for Person 3's retrieval system.

Quick start
-----------
# Person 1 hands you a ParsedDocument:
from p2_pipeline import GraphPipeline
graph = GraphPipeline().run_from_doc(parsed_doc)

# Or pass elements directly:
graph = GraphPipeline().run(parsed_doc.elements)

# Persist for Person 3:
graph.save("outputs/doc_graph.pkl")
graph.export_json("outputs/doc_graph.json")

# Person 3 retrieval calls:
graph.subgraph_around(seed_id, hops=2).all_nodes()
graph.section_context(element_id)
graph.caption_for(image_element_id)
graph.neighbors(element_id, edge_types=[EdgeType.CROSS_REF])

Integration note
----------------
Replace local imports with:
    from schema    import DocElement, ParsedDocument
    from p2_schema import EdgeType
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# from schema_p1   import DocElement, ParsedDocument  # ← Person 1's types
# from p2_schema   import EdgeType                    # ← Person 2's additions
from schema    import DocElement, ParsedDocument, DocumentEdge, EdgeType
from graph        import DocumentGraph
from heuristics   import HeuristicConfig, HeuristicGraphBuilder


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    heuristic              : HeuristicConfig = field(default_factory=HeuristicConfig)
    use_ml_similarity      : bool  = False   # needs sentence-transformers
    use_entity_cooccurrence: bool  = False   # needs spaCy
    ml_similarity_threshold: float = 0.75
    output_dir             : Optional[str] = None   # set to auto-save graph


# ── Pipeline ──────────────────────────────────────────────────────────────────

class GraphPipeline:
    """
    Orchestrates Person 2's full pipeline:

      Step 1  Heuristic graph construction   (always runs)
      Step 2  ML semantic similarity          (optional — sentence-transformers)
      Step 3  Entity co-occurrence            (optional — spaCy)
      Step 4  Persistence                     (if output_dir is set)
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.cfg = config or PipelineConfig()

    # ── Public entry points ───────────────────────────────────────────────────

    def run_from_doc(self, doc: ParsedDocument) -> DocumentGraph:
        """Convenience wrapper — accepts a ParsedDocument from Person 1."""
        return self.run(doc.elements)

    def run(self, elements: list[DocElement]) -> DocumentGraph:
        """
        Build and return a DocumentGraph from a list of DocElement objects.
        This is the method Person 3 depends on.
        """
        print(f"[P2 Pipeline] Starting with {len(elements)} elements …")

        # ── Step 1: Heuristics ────────────────────────────────────────────────
        builder = HeuristicGraphBuilder(self.cfg.heuristic)
        graph   = builder.build(elements)
        print(f"[P2 Pipeline] After heuristics: {graph}")

        # ── Step 2: ML similarity (optional) ─────────────────────────────────
        if self.cfg.use_ml_similarity:
            try:
                from ml_relations import MLRelationBuilder, MLRelationConfig
                ml = MLRelationBuilder(
                    MLRelationConfig(similarity_threshold=self.cfg.ml_similarity_threshold)
                )
                n = ml.enrich(graph)
                print(f"[P2 Pipeline] ML similarity added {n} edges.")
            except Exception as e:
                print(f"[P2 Pipeline] ML similarity skipped: {e}")

        # ── Step 3: Entity co-occurrence (optional) ───────────────────────────
        if self.cfg.use_entity_cooccurrence:
            try:
                from ml_relations import EntityCooccurrenceBuilder
                n = EntityCooccurrenceBuilder().enrich(graph)
                print(f"[P2 Pipeline] Entity co-occurrence added {n} edges.")
            except Exception as e:
                print(f"[P2 Pipeline] Entity enrichment skipped: {e}")

        # ── Step 4: Persist ───────────────────────────────────────────────────
        if self.cfg.output_dir:
            out = Path(self.cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            graph.save(out / "doc_graph.pkl")
            graph.export_json(out / "doc_graph.json")

        print(f"[P2 Pipeline] Done. {graph.stats()}")
        return graph