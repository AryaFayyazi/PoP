"""Tests for the ReasoningGraph and DSL components."""

import pytest
from pop.graph import NodeSpec, NodeType, Action, ReasoningGraph
from pop.dsl import parse_program, graph_to_program


# ---------------------------------------------------------------------------
# ReasoningGraph tests
# ---------------------------------------------------------------------------


def _make_graph():
    """Helper: build a simple 3-node graph (OCR → Det → Fuse)."""
    g = ReasoningGraph()
    g.add_node(NodeSpec(node_id="n0", node_type=NodeType.OCR, tool_key="ocr"))
    g.add_node(NodeSpec(node_id="n1", node_type=NodeType.DETECTION, tool_key="det",
                        parents=["n0"]))
    g.add_node(NodeSpec(node_id="n2", node_type=NodeType.LOGIC, is_fusion=True,
                        prompt="answer", parents=["n0", "n1"]))
    g.set_answer_node("n2")
    return g


class TestReasoningGraph:
    def test_topological_order_no_cycles(self):
        g = _make_graph()
        order = g.topological_order()
        assert order.index("n0") < order.index("n1")
        assert order.index("n1") < order.index("n2")

    def test_cycle_detection(self):
        g = ReasoningGraph()
        g.add_node(NodeSpec(node_id="a", node_type=NodeType.OCR))
        # Adding a node with an unknown parent raises ValueError
        with pytest.raises(ValueError, match="not yet in graph"):
            g.add_node(NodeSpec(node_id="b", node_type=NodeType.OCR, parents=["c"]))

    def test_duplicate_node_raises(self):
        g = _make_graph()
        with pytest.raises(ValueError, match="already exists"):
            g.add_node(NodeSpec(node_id="n0", node_type=NodeType.OCR))

    def test_set_answer_node(self):
        g = _make_graph()
        assert g.answer_node_id == "n2"

    def test_set_answer_node_unknown(self):
        g = _make_graph()
        with pytest.raises(ValueError):
            g.set_answer_node("nonexistent")

    def test_children(self):
        g = _make_graph()
        assert set(g.children("n0")) == {"n1", "n2"}
        assert g.children("n2") == []

    def test_total_cost(self):
        g = _make_graph()
        for spec in g.nodes.values():
            spec.cost = 1.0
        assert g.total_cost() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# DSL parser tests
# ---------------------------------------------------------------------------


_SAMPLE_PROGRAM = """\
CALL_TOOL(ocr, img0, "Extract text") -> v0
CALL_TOOL(det, img0[10,20,200,300], "Detect tables") -> v1
FUSE(v0, v1, "Combine and answer the question") -> v2
RETURN(v2)
"""


class TestDSLParser:
    def test_parse_basic_program(self):
        g = parse_program(_SAMPLE_PROGRAM)
        assert "v0" in g.nodes
        assert "v1" in g.nodes
        assert "v2" in g.nodes
        assert g.answer_node_id == "v2"

    def test_tool_node_fields(self):
        g = parse_program(_SAMPLE_PROGRAM)
        n0 = g.nodes["v0"]
        assert n0.tool_key == "ocr"
        assert n0.region["image_index"] == 0
        assert n0.region["bbox"] is None

    def test_bbox_region_parsed(self):
        g = parse_program(_SAMPLE_PROGRAM)
        n1 = g.nodes["v1"]
        assert n1.region["bbox"] == pytest.approx((10, 20, 200, 300))

    def test_fusion_node_fields(self):
        g = parse_program(_SAMPLE_PROGRAM)
        n2 = g.nodes["v2"]
        assert n2.is_fusion
        assert set(n2.parents) == {"v0", "v1"}

    def test_topological_order_after_parse(self):
        g = parse_program(_SAMPLE_PROGRAM)
        order = g.topological_order()
        # v0 and v1 before v2
        assert order.index("v0") < order.index("v2")
        assert order.index("v1") < order.index("v2")

    def test_missing_return_raises(self):
        prog = 'CALL_TOOL(ocr, img0, "text") -> v0\n'
        with pytest.raises(ValueError, match="RETURN"):
            parse_program(prog)

    def test_unknown_parent_in_fuse(self):
        prog = (
            'CALL_TOOL(ocr, img0, "text") -> v0\n'
            'FUSE(v0, ghost, "answer") -> v1\n'
            "RETURN(v1)\n"
        )
        with pytest.raises(ValueError):
            parse_program(prog)

    def test_round_trip(self):
        g = parse_program(_SAMPLE_PROGRAM)
        prog2 = graph_to_program(g)
        g2 = parse_program(prog2 + "\n")
        assert set(g.nodes.keys()) == set(g2.nodes.keys())
        assert g.answer_node_id == g2.answer_node_id

    def test_unrecognised_instruction_raises(self):
        prog = "INVALID_OP(foo) -> bar\nRETURN(bar)\n"
        with pytest.raises(ValueError, match="unrecognized"):
            parse_program(prog)
