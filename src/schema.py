"""
schema_p1.py
------------
This is a LOCAL COPY of Person 1's schema.py for standalone testing.
In the integrated project, delete this file and use Person 1's schema.py directly.
All imports in Person 2's code use:  from schema import DocElement, ElementType, ...
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class ElementType(str, Enum):
    HEADING   = "heading"
    PARAGRAPH = "paragraph"
    TABLE     = "table"
    IMAGE     = "image"
    CAPTION   = "caption"
    LIST_ITEM = "list_item"


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class DocElement:
    element_id:    str
    type:          ElementType
    content:       str
    page:          int
    position:      Optional[BoundingBox] = None
    heading_level: Optional[int]         = None
    parent_id:     Optional[str]         = None
    embedding:     Optional[List[float]] = None
    confidence:    Optional[float]       = None
    metadata:      Dict[str, Any]        = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class ParsedDocument:
    source_file:  str
    doc_type:     str
    total_pages:  int
    elements:     List[DocElement]   = field(default_factory=list)
    parse_stats:  Dict[str, Any]     = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            "source_file": self.source_file,
            "doc_type":    self.doc_type,
            "total_pages": self.total_pages,
            "parse_stats": self.parse_stats,
            "elements":    [e.to_dict() for e in self.elements],
        }, indent=indent, default=str)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for e in self.elements:
            counts[e.type.value] = counts.get(e.type.value, 0) + 1
        return counts

# ── Person 2 additions ────────────────────────────────────────────────────────

class EdgeType(str, Enum):
    SEQUENTIAL   = "sequential"
    PARENT_CHILD = "parent_child"
    SAME_SECTION = "same_section"
    CAPTION_OF   = "caption_of"
    FOOTNOTE_REF = "footnote_ref"
    CROSS_REF    = "cross_ref"
    CO_REFERENCE = "co_reference"
    PROXIMITY    = "proximity"


@dataclass
class DocumentEdge:
    source_id : str
    target_id : str
    edge_type : EdgeType
    weight    : float = 1.0
    metadata  : dict = field(default_factory=dict)

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