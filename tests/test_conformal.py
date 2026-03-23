"""Tests for conformal prediction utilities (§3.3–§3.4)."""

import math

import pytest
import torch

from pop.conformal import (
    CoverageTracker,
    calibrate,
    conformal_set,
    nonconformity_box,
    nonconformity_numeric,
    nonconformity_ocr,
    ConformityHead,
)
from pop.graph import NodeType


class TestNonconformityScores:
    def test_ocr_scores_in_zero_one(self):
        probs = torch.tensor([0.8, 0.1, 0.05, 0.05])
        ids = torch.tensor([0, 1, 2, 3])
        scores = nonconformity_ocr(probs, ids)
        assert scores.shape == (4,)
        assert float(scores[0]) == pytest.approx(0.2)
        assert all(0.0 <= s <= 1.0 for s in scores.tolist())

    def test_box_iou_identical_boxes(self):
        box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        scores = nonconformity_box(box.squeeze(0), box)
        assert float(scores[0]) == pytest.approx(0.0)

    def test_box_iou_no_overlap(self):
        pred = torch.tensor([0.0, 0.0, 0.5, 0.5])
        cands = torch.tensor([[0.6, 0.6, 1.0, 1.0]])
        scores = nonconformity_box(pred, cands)
        assert float(scores[0]) == pytest.approx(1.0)

    def test_numeric_residual(self):
        mean = torch.tensor(5.0)
        cands = torch.tensor([4.0, 5.0, 7.0])
        scores = nonconformity_numeric(mean, cands)
        assert scores.tolist() == pytest.approx([1.0, 0.0, 2.0])


class TestCalibrate:
    def test_perfect_calibration(self):
        # With 9 scores and δ=0.1, k = ceil(10*0.9) = 9 → last score
        scores = list(range(1, 10))  # [1, 2, ..., 9]
        tau = calibrate(scores, delta=0.1)
        # sorted: [1..9], k=ceil(10*0.9)=9, so τ = scores[8] = 9
        assert tau == 9.0

    def test_threshold_monotone_in_delta(self):
        scores = [float(i) for i in range(1, 101)]
        tau_tight = calibrate(scores, delta=0.05)
        tau_loose = calibrate(scores, delta=0.2)
        assert tau_tight >= tau_loose

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            calibrate([], delta=0.1)

    def test_single_score(self):
        tau = calibrate([0.42], delta=0.1)
        assert tau == pytest.approx(0.42)


class TestConformaSet:
    def test_all_below_threshold(self):
        candidates = ["a", "b", "c"]
        scores = [0.1, 0.2, 0.3]
        result = conformal_set(candidates, scores, threshold=0.5)
        assert result == ["a", "b", "c"]

    def test_some_filtered(self):
        candidates = ["a", "b", "c"]
        scores = [0.1, 0.6, 0.3]
        result = conformal_set(candidates, scores, threshold=0.5)
        assert "b" not in result
        assert "a" in result

    def test_kmax_truncation(self):
        candidates = list(range(10))
        scores = [float(i) * 0.05 for i in range(10)]
        result = conformal_set(candidates, scores, threshold=1.0, k_max=3)
        assert len(result) == 3
        assert result == [0, 1, 2]  # lowest scores first

    def test_empty_when_all_above(self):
        candidates = ["x", "y"]
        scores = [1.0, 2.0]
        result = conformal_set(candidates, scores, threshold=0.5)
        assert result == []

    def test_sorted_by_score(self):
        candidates = ["high", "low", "mid"]
        scores = [0.4, 0.1, 0.2]
        result = conformal_set(candidates, scores, threshold=0.5)
        assert result == ["low", "mid", "high"]


class TestConformityHead:
    def test_output_shape(self):
        head = ConformityHead(input_dim=64)
        x = torch.randn(8, 64)
        out = head(x)
        assert out.shape == (8,)

    def test_non_negative_output(self):
        head = ConformityHead(input_dim=32)
        x = torch.randn(100, 32)
        out = head(x)
        assert (out >= 0).all(), "ConformityHead must produce non-negative scores."

    def test_batched_vs_single(self):
        head = ConformityHead(input_dim=16)
        x = torch.randn(1, 16)
        out_batched = head(x)
        out_single = head(x[0].unsqueeze(0))
        assert out_batched.shape == out_single.shape


class TestCoverageTracker:
    def test_perfect_coverage(self):
        tracker = CoverageTracker(NodeType.OCR, delta=0.1)
        for i in range(10):
            tracker.update(i, list(range(10)))
        assert tracker.empirical_coverage == pytest.approx(1.0)

    def test_zero_coverage(self):
        tracker = CoverageTracker(NodeType.OCR, delta=0.1)
        for i in range(5):
            tracker.update(i, [-1])
        assert tracker.empirical_coverage == pytest.approx(0.0)

    def test_nan_when_empty(self):
        tracker = CoverageTracker(NodeType.DETECTION)
        assert math.isnan(tracker.empirical_coverage)

    def test_target_coverage(self):
        tracker = CoverageTracker(NodeType.CHART, delta=0.1)
        assert tracker.target_coverage == pytest.approx(0.9)
