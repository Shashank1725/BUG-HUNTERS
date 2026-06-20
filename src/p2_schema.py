"""
p2_schema.py  ─  Person 2 additions to the shared schema
---------------------------------------------------------
Person 1 owns: DocElement, ElementType, BoundingBox, ParsedDocument  (schema.py)
Person 2 adds: EdgeType, DocumentEdge                                 (this file)

In the integrated project every teammate imports from the shared schema.py.
Person 2's additions will be merged into that file (or kept here and imported
alongside it).  Nothing in this file redefines Person 1's types.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EdgeType(str, Enum):
    SEQUENTIAL   = "sequential"    # A appears directly before B in reading order
    PARENT_CHILD = "parent_child"  # heading → elements beneath it
    SAME_SECTION = "same_section"  # siblings sharing the same heading ancestor
    CAPTION_OF   = "caption_of"   # caption describes an image or table
    FOOTNOTE_REF = "footnote_ref"  # paragraph references a footnote
    CROSS_REF    = "cross_ref"    # explicit "see Table 3" / "Figure 2" mention
    CO_REFERENCE = "co_reference"  # ML: shared entity / topic
    PROXIMITY    = "proximity"     # spatially close elements on the same page


@dataclass
class DocumentEdge:
    """
    A directed, typed, weighted relationship between two DocElement IDs.

    source_id / target_id  ──  match DocElement.element_id from Person 1's schema
    """
    source_id : str
    target_id : str
    edge_type : EdgeType
    weight    : float = 1.0
    metadata  : dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id" : self.source_id,
            "target_id" : self.target_id,
            "edge_type" : self.edge_type.value,
            "weight"    : self.weight,
            "metadata"  : self.metadata,
        }

    def __repr__(self) -> str:
        return (f"<Edge {self.edge_type.value}: "
                f"{self.source_id} → {self.target_id} (w={self.weight:.2f})>")