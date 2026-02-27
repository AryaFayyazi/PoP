"""End-to-end inference tests for the PoP model (§3.8)."""

import pytest

from pop import ProofOfPerception
from pop.graph import NodeType
from pop.dsl import parse_program


class TestProofOfPerception:
    def _make_model(self):
        return ProofOfPerception(
            feature_dim=32,
            context_dim=32,
            budget=16.0,
            delta=0.1,
        )

    def test_infer_returns_set_and_graph(self):
        model = self._make_model()
        answer_set, graph = model.infer(images=[], query="What is the total?")
        assert isinstance(answer_set, list)
        assert graph.answer_node_id is not None

    def test_infer_with_custom_planner(self):
        model = self._make_model()
        program = (
            'CALL_TOOL(ocr, img0, "read") -> n0\n'
            'FUSE(n0, "answer") -> n1\n'
            "RETURN(n1)\n"
        )
        answer_set, graph = model.infer(
            images=[], query="q", planner=lambda imgs, q: program
        )
        assert "n0" in graph.nodes
        assert "n1" in graph.nodes

    def test_calibrate_nodes(self):
        model = self._make_model()
        # Add some calibration scores for OCR
        for score in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            model.add_calibration_score(NodeType.OCR, score)
        updated = model.calibrate_nodes(delta=0.1)
        assert "ocr-string" in updated
        assert 0.0 < updated["ocr-string"] <= 1.0

    def test_calibrate_skips_empty_pools(self):
        model = self._make_model()
        # No scores added for any type
        updated = model.calibrate_nodes()
        assert updated == {}

    def test_budget_respected(self):
        # Very tight budget: only 1 unit, each tool call costs 1
        model = self._make_model()
        model.budget = 1.0
        answer_set, graph = model.infer(images=[], query="test")
        # Should not crash; budget may cause early stop
        assert isinstance(answer_set, list)

    def test_planner_fallback_on_invalid_program(self):
        model = self._make_model()
        # Planner returns an invalid program → should fall back to default
        answer_set, graph = model.infer(
            images=[], query="q", planner=lambda imgs, q: "INVALID"
        )
        assert graph.answer_node_id is not None

    def test_calibration_affects_thresholds(self):
        model = self._make_model()
        initial = dict(model.thresholds)
        for score in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
            model.add_calibration_score(NodeType.DETECTION, score)
        model.calibrate_nodes()
        # Detection threshold should now differ from initial (1.0)
        assert model.thresholds["det-box"] != initial["det-box"]
