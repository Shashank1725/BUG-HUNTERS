"""
ml_relations.py  ─  Optional ML-based relationship enrichment
-------------------------------------------------------------
Adds CO_REFERENCE edges between DocElements that discuss the same
entity or topic.  Two independent enrichers are provided:

  MLRelationBuilder          cosine similarity via sentence-transformers
  EntityCooccurrenceBuilder  named-entity co-occurrence via spaCy

Both are ADDITIVE — they supplement the heuristic graph, never replace it.
Both accept a DocumentGraph and mutate it in-place.

Dependencies (install separately — not required for core pipeline):
    pip install sentence-transformers
    pip install spacy && python -m spacy download en_core_web_sm

Integration note
----------------
Replace local imports with:
    from schema    import DocElement, ElementType
    from p2_schema import DocumentEdge, EdgeType
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

# from schema_p1 import DocElement, ElementType   # ← Person 1's types
# from p2_schema  import DocumentEdge, EdgeType   # ← Person 2's additions
from schema import DocElement, ElementType, DocumentEdge, EdgeType  # ← shared repo
from graph       import DocumentGraph

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Semantic similarity  (sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MLRelationConfig:
    similarity_threshold : float = 0.75
    eligible_types       : tuple = (
        ElementType.PARAGRAPH,
        ElementType.HEADING,
        ElementType.CAPTION,
    )
    embedding_model      : str   = "all-MiniLM-L6-v2"
    max_nodes            : int   = 500   # O(n²) — cap for large documents


class MLRelationBuilder:
    """
    Adds CO_REFERENCE edges between elements whose sentence-transformer
    embeddings exceed `similarity_threshold` in cosine similarity.

    Usage
    -----
    ml = MLRelationBuilder()
    n_added = ml.enrich(graph)
    """

    def __init__(self, config: Optional[MLRelationConfig] = None) -> None:
        self.cfg    = config or MLRelationConfig()
        self._model = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.cfg.embedding_model)
            logger.info("[MLRelations] loaded model '%s'", self.cfg.embedding_model)
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Run:  pip install sentence-transformers"
            )

    # ── Public ────────────────────────────────────────────────────────────────

    def enrich(self, graph: DocumentGraph) -> int:
        """
        Add CO_REFERENCE edges to `graph`.
        Returns the number of edges added.
        """
        self._load_model()
        from sentence_transformers import util  # type: ignore

        eligible = [
            n for n in graph.all_nodes()
            if n.type in self.cfg.eligible_types and n.content.strip()
        ][: self.cfg.max_nodes]

        if len(eligible) < 2:
            return 0

        texts      = [n.content[:512] for n in eligible]
        embeddings = self._model.encode(texts, convert_to_tensor=True,  # type: ignore
                                        show_progress_bar=False)
        scores     = util.cos_sim(embeddings, embeddings)  # type: ignore
        added      = 0

        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                score = float(scores[i][j])
                if score >= self.cfg.similarity_threshold:
                    graph.add_edge(DocumentEdge(
                        source_id=eligible[i].element_id,
                        target_id=eligible[j].element_id,
                        edge_type=EdgeType.CO_REFERENCE,
                        weight=round(score, 4),
                        metadata={"similarity": round(score, 4)},
                    ))
                    added += 1

        logger.info("[MLRelations] added %d CO_REFERENCE edges (similarity)", added)
        return added


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Named-entity co-occurrence  (spaCy)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityCooccurrenceConfig:
    spacy_model    : str   = "en_core_web_sm"
    entity_labels  : set   = field(default_factory=lambda: {
        "PERSON", "ORG", "GPE", "PRODUCT", "LAW", "EVENT"
    })
    edge_weight    : float = 0.7
    content_limit  : int   = 1000   # characters to feed spaCy per element


class EntityCooccurrenceBuilder:
    """
    Uses spaCy NER to find elements that mention the same named entity
    and adds CO_REFERENCE edges between them.

    Usage
    -----
    ent = EntityCooccurrenceBuilder()
    n_added = ent.enrich(graph)
    """

    def __init__(self, config: Optional[EntityCooccurrenceConfig] = None) -> None:
        self.cfg  = config or EntityCooccurrenceConfig()
        self._nlp = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._nlp:
            return
        try:
            import spacy
            self._nlp = spacy.load(self.cfg.spacy_model)
        except (ImportError, OSError) as e:
            raise RuntimeError(
                f"spaCy model '{self.cfg.spacy_model}' not available.\n"
                "Run:  python -m spacy download en_core_web_sm"
            ) from e

    # ── Public ────────────────────────────────────────────────────────────────

    def enrich(self, graph: DocumentGraph) -> int:
        """
        Add CO_REFERENCE edges to `graph`.
        Returns the number of edges added.
        """
        self._load()
        nodes = [n for n in graph.all_nodes() if n.content.strip()]

        # entity_key → [element_ids that mention it]
        entity_map: dict[str, list[str]] = {}
        for node in nodes:
            doc = self._nlp(node.content[: self.cfg.content_limit])  # type: ignore
            for ent in doc.ents:
                if ent.label_ in self.cfg.entity_labels:
                    key = f"{ent.label_}:{ent.text.lower()}"
                    entity_map.setdefault(key, []).append(node.element_id)

        added = 0
        for entity, ids in entity_map.items():
            unique_ids = list(dict.fromkeys(ids))   # deduplicate while preserving order
            for i in range(len(unique_ids)):
                for j in range(i + 1, len(unique_ids)):
                    graph.add_edge(DocumentEdge(
                        source_id=unique_ids[i],
                        target_id=unique_ids[j],
                        edge_type=EdgeType.CO_REFERENCE,
                        weight=self.cfg.edge_weight,
                        metadata={"entity": entity},
                    ))
                    added += 1

        logger.info("[EntityCooccurrence] added %d CO_REFERENCE edges", added)
        return added