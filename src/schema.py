"""
schema.py
---------
Unified output schema for every extracted document element.
Every parser (PDF / DOCX) emits elements in this exact shape so
downstream consumers (RAG indexer, LLM context builder, etc.)
never need to know which parser produced the data.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    LIST_ITEM = "list_item"


@dataclass
class BoundingBox:
    """Position of the element on the page (PDF coordinate space)."""
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class DocElement:
    """
    A single structured unit extracted from a document.
    This is the ATOMIC output of the ingestion pipeline.
    """
    element_id: str                       # unique id, e.g. "doc1_p3_el7"
    type: ElementType
    content: str                          # text content / table-as-markdown / image caption
    page: int                             # 1-indexed page number
    position: Optional[BoundingBox] = None
    heading_level: Optional[int] = None   # 1-6 for headings, None otherwise
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class ParsedDocument:
    """Top-level container for a fully parsed document."""
    source_file: str
    doc_type: str                         # "pdf" or "docx"
    total_pages: int
    elements: List[DocElement] = field(default_factory=list)
    parse_stats: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            "source_file": self.source_file,
            "doc_type": self.doc_type,
            "total_pages": self.total_pages,
            "parse_stats": self.parse_stats,
            "elements": [e.to_dict() for e in self.elements],
        }, indent=indent, default=str)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())

    def summary(self) -> dict:
        counts = {}
        for e in self.elements:
            counts[e.type.value] = counts.get(e.type.value, 0) + 1
        return counts
