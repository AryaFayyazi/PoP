"""
Tests for pop/nodes – BaseNode subclasses and NodeRegistry.

Coverage
--------
- OCRNode: execute returns a list; conformal filtering; MAP fallback.
- DetectionNode: execute returns box tuples; conformal filtering; MAP fallback.
- ChartNode: execute returns numeric values; conformal filtering; MAP fallback.
- LogicFusionNode: execute returns text strings; cert_head shapes; parent context.
- NodeRegistry: build, get, register; shared instances; unknown key.
"""

import pytest
import torch

from pop.graph import NodeSpec, NodeType
from pop.nodes import (
    BaseNode,
    ChartNode,
    DetectionNode,
    LogicFusionNode,
    NodeRegistry,
    OCRNode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ocr_spec(nid: str = "v0") -> NodeSpec:
    return NodeSpec(
        node_id=nid,
        node_type=NodeType.OCR,
        tool_key="ocr",
        region={"image_index": 0, "bbox": None},
        prompt="Extract text",
        is_fusion=False,
        parents=[],
    )


def _det_spec(nid: str = "v1") -> NodeSpec:
    return NodeSpec(
        node_id=nid,
        node_type=NodeType.DETECTION,
        tool_key="det",
        region={"image_index": 0, "bbox": None},
        prompt="Detect objects",
        is_fusion=False,
        parents=[],
    )


def _chart_spec(nid: str = "v2") -> NodeSpec:
    return NodeSpec(
        node_id=nid,
        node_type=NodeType.CHART,
        tool_key="chart",
        region={"image_index": 0, "bbox": None},
        prompt="Read chart value",
        is_fusion=False,
        parents=[],
    )


def _fuse_spec(nid: str = "v3", parents=("v0",)) -> NodeSpec:
    return NodeSpec(
        node_id=nid,
        node_type=NodeType.LOGIC,
        is_fusion=True,
        prompt="Answer the question",
        parents=list(parents),
    )


def _context(query: str = "What is the total?", **parent_sets) -> dict:
    ctx = {"__query__": query}
    ctx.update(parent_sets)
    return ctx


# ---------------------------------------------------------------------------
# BaseNode: abstract interface
# ---------------------------------------------------------------------------


class TestBaseNode:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseNode()  # type: ignore[abstract]

    def test_all_concrete_nodes_are_subclasses(self):
        for cls in (OCRNode, DetectionNode, ChartNode, LogicFusionNode):
            assert issubclass(cls, BaseNode)


# ---------------------------------------------------------------------------
# OCRNode
# ---------------------------------------------------------------------------


class TestOCRNode:
    def _make(self, feature_dim: int = 32) -> OCRNode:
        return OCRNode(feature_dim=feature_dim)

    def test_execute_returns_list(self):
        node = self._make()
        result = node.execute(_ocr_spec(), images=[], threshold=1.0, context=_context())
        assert isinstance(result, list)

    def test_execute_non_empty(self):
        node = self._make()
        result = node.execute(_ocr_spec(), images=[], threshold=1.0, context=_context())
        assert len(result) > 0

    def test_execute_items_are_strings(self):
        node = self._make()
        result = node.execute(_ocr_spec(), images=[], threshold=1.0, context=_context())
        for item in result:
            assert isinstance(item, str)

    def test_threshold_zero_returns_only_map(self):
        """With threshold=0, only scores ≤ 0 qualify; s^(ocr) for MAP = 1−0.6 = 0.4 > 0,
        so the set would be empty → fallback to MAP candidate."""
        node = self._make()
        result = node.execute(_ocr_spec(), images=[], threshold=0.0, context=_context())
        assert len(result) == 1  # fallback to MAP

    def test_threshold_one_returns_all_candidates(self):
        """With threshold=1.0, all candidates qualify (all scores are in [0, 1])."""
        node = self._make()
        result = node.execute(_ocr_spec(), images=[], threshold=1.0, context=_context())
        assert len(result) == OCRNode.K_MAX

    def test_respects_k_max(self):
        node = self._make()
        result = node.execute(_ocr_spec(), images=[], threshold=1.0, context=_context())
        assert len(result) <= OCRNode.K_MAX

    def test_cert_head_is_present(self):
        node = self._make(feature_dim=64)
        assert hasattr(node, "cert_head")

    def test_cert_head_output_shape(self):
        node = self._make(feature_dim=64)
        feat = torch.randn(3, 64)
        scores = node.cert_head(feat)
        assert scores.shape == (3,)
        assert (scores >= 0).all()  # Softplus → non-negative


# ---------------------------------------------------------------------------
# DetectionNode
# ---------------------------------------------------------------------------


class TestDetectionNode:
    def _make(self, feature_dim: int = 32) -> DetectionNode:
        return DetectionNode(feature_dim=feature_dim)

    def test_execute_returns_list(self):
        node = self._make()
        result = node.execute(_det_spec(), images=[], threshold=1.0, context=_context())
        assert isinstance(result, list)

    def test_execute_non_empty(self):
        node = self._make()
        result = node.execute(_det_spec(), images=[], threshold=1.0, context=_context())
        assert len(result) > 0

    def test_items_are_tuples_of_four_floats(self):
        node = self._make()
        result = node.execute(_det_spec(), images=[], threshold=1.0, context=_context())
        for box in result:
            assert isinstance(box, tuple)
            assert len(box) == 4

    def test_map_box_always_included_at_threshold_zero(self):
        """MAP box has nonconformity score 0 (IoU=1 with itself) → always included."""
        node = self._make()
        result = node.execute(_det_spec(), images=[], threshold=0.0, context=_context())
        # MAP box is (10, 10, 100, 100)
        assert (10.0, 10.0, 100.0, 100.0) in result

    def test_threshold_low_reduces_set_size(self):
        """A very tight threshold should include fewer boxes."""
        node = self._make()
        r_tight = node.execute(_det_spec(), images=[], threshold=0.05, context=_context())
        r_loose = node.execute(_det_spec(), images=[], threshold=0.90, context=_context())
        assert len(r_tight) <= len(r_loose)

    def test_respects_k_max(self):
        node = self._make()
        result = node.execute(_det_spec(), images=[], threshold=1.0, context=_context())
        assert len(result) <= DetectionNode.K_MAX

    def test_cert_head_output_non_negative(self):
        node = self._make(feature_dim=32)
        feat = torch.randn(2, 32)
        scores = node.cert_head(feat)
        assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# ChartNode
# ---------------------------------------------------------------------------


class TestChartNode:
    def _make(self, feature_dim: int = 32) -> ChartNode:
        return ChartNode(feature_dim=feature_dim)

    def test_execute_returns_list(self):
        node = self._make()
        result = node.execute(_chart_spec(), images=[], threshold=2.0, context=_context())
        assert isinstance(result, list)

    def test_execute_non_empty(self):
        node = self._make()
        result = node.execute(_chart_spec(), images=[], threshold=2.0, context=_context())
        assert len(result) > 0

    def test_items_are_numeric(self):
        node = self._make()
        result = node.execute(_chart_spec(), images=[], threshold=2.0, context=_context())
        for val in result:
            assert isinstance(val, (int, float))

    def test_map_value_always_in_result(self):
        """The MAP value has nonconformity score = 0 (|42 − 42| = 0), so it
        is included for any threshold ≥ 0."""
        node = self._make()
        result = node.execute(_chart_spec(), images=[], threshold=0.0, context=_context())
        # MAP value is 42.0 (score = |42 − 42| = 0)
        assert 42.0 in result

    def test_threshold_controls_set_width(self):
        """Higher threshold → wider interval (more candidates)."""
        node = self._make()
        r_narrow = node.execute(_chart_spec(), images=[], threshold=0.5, context=_context())
        r_wide   = node.execute(_chart_spec(), images=[], threshold=5.0, context=_context())
        assert len(r_narrow) <= len(r_wide)

    def test_respects_k_max(self):
        node = self._make()
        result = node.execute(_chart_spec(), images=[], threshold=100.0, context=_context())
        assert len(result) <= ChartNode.K_MAX


# ---------------------------------------------------------------------------
# LogicFusionNode
# ---------------------------------------------------------------------------


class TestLogicFusionNode:
    def _make(self, feature_dim: int = 32) -> LogicFusionNode:
        return LogicFusionNode(feature_dim=feature_dim)

    def test_execute_returns_list(self):
        node = self._make()
        result = node.execute(_fuse_spec(), images=[], threshold=10.0, context=_context())
        assert isinstance(result, list)

    def test_execute_non_empty(self):
        node = self._make()
        result = node.execute(_fuse_spec(), images=[], threshold=10.0, context=_context())
        assert len(result) > 0

    def test_items_are_strings(self):
        node = self._make()
        result = node.execute(_fuse_spec(), images=[], threshold=10.0, context=_context())
        for item in result:
            assert isinstance(item, str)

    def test_respects_k_max(self):
        node = self._make()
        result = node.execute(_fuse_spec(), images=[], threshold=100.0, context=_context())
        assert len(result) <= LogicFusionNode.K_MAX

    def test_fallback_when_threshold_zero(self):
        """At threshold=0 all ConformityHead scores > 0 → empty → fallback."""
        node = self._make()
        result = node.execute(_fuse_spec(), images=[], threshold=0.0, context=_context())
        assert len(result) == 1  # fallback to best candidate

    def test_uses_parent_context(self):
        """Execute should not raise when parent node sets are provided."""
        node = self._make()
        ctx = _context(query="q", v0=["some text"], v1=["other text"])
        spec = _fuse_spec(parents=["v0", "v1"])
        result = node.execute(spec, images=[], threshold=10.0, context=ctx)
        assert isinstance(result, list)

    def test_cert_head_shape(self):
        node = self._make(feature_dim=32)
        feat = torch.randn(5, 32)
        scores = node.cert_head(feat)
        assert scores.shape == (5,)
        assert (scores >= 0).all()

    def test_input_proj_shape(self):
        node = self._make(feature_dim=64)
        x = torch.randn(3, 64)
        out = node.input_proj(x)
        assert out.shape == (3, 64)


# ---------------------------------------------------------------------------
# NodeRegistry
# ---------------------------------------------------------------------------


class TestNodeRegistry:
    def _registry(self, feature_dim: int = 32) -> NodeRegistry:
        return NodeRegistry.build(feature_dim=feature_dim)

    def test_build_returns_registry(self):
        reg = self._registry()
        assert isinstance(reg, NodeRegistry)

    def test_get_ocr(self):
        reg = self._registry()
        assert isinstance(reg.get("ocr"), OCRNode)

    def test_get_det(self):
        reg = self._registry()
        assert isinstance(reg.get("det"), DetectionNode)

    def test_get_detector_alias(self):
        reg = self._registry()
        assert isinstance(reg.get("detector"), DetectionNode)

    def test_get_chart(self):
        reg = self._registry()
        assert isinstance(reg.get("chart"), ChartNode)

    def test_get_fuse(self):
        reg = self._registry()
        assert isinstance(reg.get("fuse"), LogicFusionNode)

    def test_get_layout(self):
        reg = self._registry()
        assert isinstance(reg.get("layout"), LogicFusionNode)

    def test_get_vqa(self):
        reg = self._registry()
        assert isinstance(reg.get("vqa"), LogicFusionNode)

    def test_get_unknown_returns_none(self):
        reg = self._registry()
        assert reg.get("nonexistent_tool") is None

    def test_shared_ocr_instances(self):
        """'ocr' key must share the same instance as other OCR keys (none here,
        but the design principle is tested via det / detector)."""
        reg = self._registry()
        assert reg.get("det") is reg.get("detector")

    def test_shared_fuse_instances(self):
        reg = self._registry()
        assert reg.get("fuse") is reg.get("layout")
        assert reg.get("fuse") is reg.get("vqa")

    def test_register_custom_node(self):
        reg = self._registry()
        custom = OCRNode(feature_dim=16)
        reg.register("custom_ocr", custom)
        assert reg.get("custom_ocr") is custom

    def test_repr_contains_keys(self):
        reg = self._registry()
        r = repr(reg)
        assert "ocr" in r
        assert "det" in r
        assert "chart" in r
        assert "fuse" in r

    def test_execute_via_registry(self):
        """Round-trip: get a node from registry and run execute."""
        reg = self._registry()
        node = reg.get("ocr")
        assert node is not None
        result = node.execute(
            _ocr_spec(), images=[], threshold=1.0, context=_context()
        )
        assert isinstance(result, list)
        assert len(result) > 0
