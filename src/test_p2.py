"""
test_p2.py  ─  Full test suite for Person 2
--------------------------------------------
Run with:  python test_p2.py
All tests use only stdlib + networkx (no ML deps required).
"""

import sys
import unittest

# from schema_p1  import DocElement, ElementType, BoundingBox, ParsedDocument
# from p2_schema  import DocumentEdge, EdgeType
from schema import DocElement, ElementType, BoundingBox, ParsedDocument, DocumentEdge, EdgeType
from graph       import DocumentGraph
from heuristics  import HeuristicGraphBuilder, HeuristicConfig
from p2_pipeline import GraphPipeline, PipelineConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def el(
    eid: str,
    etype: ElementType,
    content: str,
    page: int = 1,
    x0=0.0, y0=0.0, x1=200.0, y1=20.0,
    heading_level: int = None,
    **meta,
) -> DocElement:
    return DocElement(
        element_id=eid,
        type=etype,
        content=content,
        page=page,
        position=BoundingBox(x0, y0, x1, y1),
        heading_level=heading_level,
        metadata=meta,
    )


def sample_elements() -> list[DocElement]:
    return [
        el("h1",   ElementType.HEADING,   "Introduction",        page=1, y0=0,   y1=20,  heading_level=1),
        el("p1",   ElementType.PARAGRAPH, "This paper covers…",  page=1, y0=30,  y1=50),
        el("p2",   ElementType.PARAGRAPH, "See Figure 1 …",      page=1, y0=60,  y1=80),
        el("img1", ElementType.IMAGE,     "[image]",             page=1, y0=90,  y1=160),
        el("cap1", ElementType.CAPTION,   "Figure 1: Results",   page=1, y0=165, y1=180),
        el("fn1",  ElementType.PARAGRAPH, "[1] Full citation.",  page=1, y0=400, y1=415, is_footnote=True),
        el("h2",   ElementType.HEADING,   "Methods",             page=2, y0=0,   y1=20,  heading_level=1),
        el("p3",   ElementType.PARAGRAPH, "We used Python…",     page=2, y0=30,  y1=50),
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSchema(unittest.TestCase):

    def test_edge_type_values(self):
        self.assertEqual(EdgeType.SEQUENTIAL.value, "sequential")
        self.assertEqual(EdgeType.CAPTION_OF.value, "caption_of")

    def test_document_edge_repr(self):
        e = DocumentEdge("a", "b", EdgeType.CROSS_REF, weight=0.9)
        self.assertIn("cross_ref", repr(e))
        self.assertIn("a → b", repr(e))

    def test_document_edge_to_dict(self):
        e = DocumentEdge("x", "y", EdgeType.PROXIMITY, weight=0.5, metadata={"d": 10})
        d = e.to_dict()
        self.assertEqual(d["edge_type"], "proximity")
        self.assertEqual(d["metadata"]["d"], 10)


class TestDocumentGraph(unittest.TestCase):

    def _build(self) -> DocumentGraph:
        cfg = HeuristicConfig(
            sequential_gap_threshold=200,
            caption_proximity_threshold=60,
            proximity_distance_threshold=150,
        )
        return HeuristicGraphBuilder(cfg).build(sample_elements())

    def test_node_count(self):
        g = self._build()
        self.assertEqual(len(g.all_nodes()), 8)

    def test_get_node(self):
        g = self._build()
        node = g.get_node("h1")
        self.assertIsNotNone(node)
        self.assertEqual(node.type, ElementType.HEADING)

    def test_get_node_missing(self):
        g = self._build()
        self.assertIsNone(g.get_node("does_not_exist"))

    def test_nodes_by_type(self):
        g = self._build()
        headings = g.nodes_by_type(ElementType.HEADING)
        self.assertEqual(len(headings), 2)

    def test_nodes_on_page(self):
        g = self._build()
        self.assertEqual(len(g.nodes_on_page(1)), 6)
        self.assertEqual(len(g.nodes_on_page(2)), 2)

    def test_stats_keys(self):
        g = self._build()
        s = g.stats()
        for key in ("total_nodes", "total_edges", "node_types", "edge_types"):
            self.assertIn(key, s)

    def test_stats_counts(self):
        g = self._build()
        s = g.stats()
        self.assertGreater(s["total_nodes"], 0)
        self.assertGreater(s["total_edges"], 0)


class TestSequentialEdges(unittest.TestCase):

    def test_h1_to_p1_sequential(self):
        g = HeuristicGraphBuilder(
            HeuristicConfig(sequential_gap_threshold=200)
        ).build(sample_elements())
        nbrs = [n.element_id for n in g.neighbors("h1", [EdgeType.SEQUENTIAL], "out")]
        self.assertIn("p1", nbrs)

    def test_large_gap_breaks_sequential(self):
        # fn1 is at y0=400, img1 ends at y1=160 → gap=240 > threshold=200
        g = HeuristicGraphBuilder(
            HeuristicConfig(sequential_gap_threshold=200)
        ).build(sample_elements())
        # cap1 (y1=180) → fn1 (y0=400) gap=220 > 200, so no sequential edge
        nbrs = [n.element_id for n in g.neighbors("cap1", [EdgeType.SEQUENTIAL], "out")]
        self.assertNotIn("fn1", nbrs)


class TestHeadingSectionEdges(unittest.TestCase):

    def test_parent_child_h1_p1(self):
        g = HeuristicGraphBuilder().build(sample_elements())
        children = [n.element_id for n in g.neighbors("h1", [EdgeType.PARENT_CHILD], "out")]
        self.assertIn("p1", children)
        self.assertIn("p2", children)

    def test_same_section_p1_p2(self):
        g = HeuristicGraphBuilder().build(sample_elements())
        sibs = [n.element_id for n in g.neighbors("p1", [EdgeType.SAME_SECTION])]
        self.assertIn("p2", sibs)

    def test_section_context_includes_heading(self):
        g = HeuristicGraphBuilder().build(sample_elements())
        ctx_ids = [n.element_id for n in g.section_context("p1")]
        self.assertIn("h1", ctx_ids)

    def test_h2_children_on_page2(self):
        g = HeuristicGraphBuilder().build(sample_elements())
        children = [n.element_id for n in g.neighbors("h2", [EdgeType.PARENT_CHILD], "out")]
        self.assertIn("p3", children)
        # p1, p2 should NOT be children of h2
        self.assertNotIn("p1", children)


class TestCaptionEdges(unittest.TestCase):

    def test_caption_of_img1(self):
        g = HeuristicGraphBuilder(
            HeuristicConfig(caption_proximity_threshold=60)
        ).build(sample_elements())
        linked = [n.element_id for n in g.neighbors("cap1", [EdgeType.CAPTION_OF], "out")]
        self.assertIn("img1", linked)

    def test_caption_for_helper(self):
        g = HeuristicGraphBuilder(
            HeuristicConfig(caption_proximity_threshold=60)
        ).build(sample_elements())
        cap = g.caption_for("img1")
        self.assertIsNotNone(cap)
        self.assertEqual(cap.element_id, "cap1")

    def test_no_cross_page_caption(self):
        els = [
            el("img_p1",  ElementType.IMAGE,   "[img]",           page=1, y0=50, y1=100),
            el("cap_p2",  ElementType.CAPTION, "Figure 1: desc",  page=2, y0=10, y1=25),
        ]
        g = HeuristicGraphBuilder().build(els)
        linked = g.neighbors("cap_p2", [EdgeType.CAPTION_OF], "out")
        self.assertEqual(len(linked), 0)


class TestCrossReferenceEdges(unittest.TestCase):

    def test_see_figure_1(self):
        els = [
            el("p_ref",   ElementType.PARAGRAPH, "Results are shown in see Figure 1.", page=1, y0=0,  y1=20),
            el("cap_fig", ElementType.CAPTION,   "Figure 1: Performance",              page=1, y0=30, y1=50),
        ]
        g = HeuristicGraphBuilder().build(els)
        xrefs = [n.element_id for n in g.neighbors("p_ref", [EdgeType.CROSS_REF], "out")]
        self.assertIn("cap_fig", xrefs)

    def test_refer_to_table(self):
        els = [
            el("p_tbl",   ElementType.PARAGRAPH, "As shown, refer to Table 2 for details.", page=1, y0=0,  y1=20),
            el("cap_tbl", ElementType.CAPTION,   "Table 2: Statistics",                     page=1, y0=30, y1=50),
        ]
        g = HeuristicGraphBuilder().build(els)
        xrefs = [n.element_id for n in g.neighbors("p_tbl", [EdgeType.CROSS_REF], "out")]
        self.assertIn("cap_tbl", xrefs)


class TestFootnoteEdges(unittest.TestCase):

    def test_footnote_ref_linked(self):
        els = [
            el("p_fn",  ElementType.PARAGRAPH, "Some claim [1] needs citation.", page=1, y0=0, y1=20),
            el("fn_el", ElementType.PARAGRAPH, "[1] Author et al. 2024",         page=1, y0=400, y1=415, is_footnote=True),
        ]
        g = HeuristicGraphBuilder().build(els)
        linked = [n.element_id for n in g.neighbors("p_fn", [EdgeType.FOOTNOTE_REF], "out")]
        self.assertIn("fn_el", linked)


class TestSubgraphAround(unittest.TestCase):

    def test_subgraph_contains_seed(self):
        g = HeuristicGraphBuilder(HeuristicConfig(sequential_gap_threshold=200)).build(sample_elements())
        sub = g.subgraph_around("p1", hops=1)
        ids = {n.element_id for n in sub.all_nodes()}
        self.assertIn("p1", ids)

    def test_subgraph_larger_than_seed(self):
        g = HeuristicGraphBuilder(HeuristicConfig(sequential_gap_threshold=200)).build(sample_elements())
        sub = g.subgraph_around("p1", hops=1)
        self.assertGreater(len(sub.all_nodes()), 1)

    def test_subgraph_hops_0(self):
        g = HeuristicGraphBuilder().build(sample_elements())
        sub = g.subgraph_around("p1", hops=0)
        ids = {n.element_id for n in sub.all_nodes()}
        self.assertEqual(ids, {"p1"})


class TestPersistence(unittest.TestCase):

    def test_save_and_load(self):
        g    = HeuristicGraphBuilder().build(sample_elements())
        path = "/tmp/test_doc_graph.pkl"
        g.save(path)
        g2 = DocumentGraph.load(path)
        self.assertEqual(len(g2.all_nodes()), len(g.all_nodes()))
        self.assertEqual(g2.stats()["total_edges"], g.stats()["total_edges"])

    def test_export_json(self, path="/tmp/test_doc_graph.json"):
        import json
        g = HeuristicGraphBuilder().build(sample_elements())
        g.export_json(path)
        data = json.loads(open(path).read())
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertEqual(len(data["nodes"]), len(g.all_nodes()))


class TestPipeline(unittest.TestCase):

    def test_run_returns_graph(self):
        g = GraphPipeline().run(sample_elements())
        self.assertIsInstance(g, DocumentGraph)

    def test_run_from_doc(self):
        doc = ParsedDocument(
            source_file="test.pdf",
            doc_type="pdf",
            total_pages=2,
            elements=sample_elements(),
        )
        g = GraphPipeline().run_from_doc(doc)
        self.assertEqual(len(g.all_nodes()), 8)

    def test_pipeline_with_output_dir(self):
        import os, json
        cfg = PipelineConfig(output_dir="/tmp/p2_test_out")
        GraphPipeline(cfg).run(sample_elements())
        self.assertTrue(os.path.exists("/tmp/p2_test_out/doc_graph.pkl"))
        self.assertTrue(os.path.exists("/tmp/p2_test_out/doc_graph.json"))

    def test_ml_flags_off_by_default(self):
        cfg = PipelineConfig()
        self.assertFalse(cfg.use_ml_similarity)
        self.assertFalse(cfg.use_entity_cooccurrence)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)