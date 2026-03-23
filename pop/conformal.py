"""
Conformal prediction utilities for PoP (§3.3–§3.4).

For each node type t, PoP learns a nonconformity function
    s^(t)(x_v, z) : X_v × Z_v → R_{≥0}
and uses split-conformal calibration to select a threshold τ^(t)_δ such that
    P(z_true ∈ Γ^(t)_δ(x_v)) ≥ 1 − δ     (marginal coverage guarantee).

This module provides:
- ``nonconformity_score`` – default score families per NodeType
- ``calibrate`` – split-CP threshold selection (§3.4, Eq. 15)
- ``conformal_set`` – set-valued prediction (§3.4, Eq. 16)
- ``ConformityHead`` – a trainable certificate head (§3.4, Eq. 18)
- ``CoverageTracker`` – running empirical coverage monitor
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from pop.graph import NodeType

# ---------------------------------------------------------------------------
# Default nonconformity score functions (closed-form, no parameters)
# ---------------------------------------------------------------------------


def nonconformity_ocr(
    model_probs: torch.Tensor,
    candidate_string_ids: torch.Tensor,
) -> torch.Tensor:
    """
    s^(ocr)(x_v, z) = 1 − P_θ(z | x_v)   (Eq. 9)

    Parameters
    ----------
    model_probs:
        Tensor [C] of token or string probabilities from the base model.
    candidate_string_ids:
        Tensor [K] of candidate indices into ``model_probs``.

    Returns
    -------
    Tensor [K] of nonconformity scores in [0, 1].
    """
    probs = model_probs[candidate_string_ids].clamp(min=0.0, max=1.0)
    return 1.0 - probs


def nonconformity_box(
    pred_box: torch.Tensor,
    candidate_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    s^(box)(x_v, z) = 1 − IoU(z, ẑ_MAP)   (Eq. 10)

    Parameters
    ----------
    pred_box:
        Tensor [4] (x1, y1, x2, y2) – MAP (highest-scoring) box.
    candidate_boxes:
        Tensor [K, 4].

    Returns
    -------
    Tensor [K] of nonconformity scores in [0, 1].
    """
    iou = _box_iou(pred_box.unsqueeze(0), candidate_boxes)  # [K]
    return (1.0 - iou).clamp(min=0.0, max=1.0)


def nonconformity_numeric(
    pred_mean: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """
    s^(num)(x_v, z) = |z − μ_θ(x_v)|   (Eq. 11)

    Parameters
    ----------
    pred_mean:
        Scalar tensor – predicted mean.
    candidates:
        Tensor [K] of candidate numeric values.

    Returns
    -------
    Tensor [K] of non-negative residuals.
    """
    return (candidates - pred_mean).abs()


def _box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between each pair (a_i, b_i).

    Parameters
    ----------
    boxes_a, boxes_b : Tensor [N, 4]

    Returns
    -------
    Tensor [N] of IoU values.
    """
    inter_x1 = torch.max(boxes_a[:, 0], boxes_b[:, 0])
    inter_y1 = torch.max(boxes_a[:, 1], boxes_b[:, 1])
    inter_x2 = torch.min(boxes_a[:, 2], boxes_b[:, 2])
    inter_y2 = torch.min(boxes_a[:, 3], boxes_b[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union_area = area_a + area_b - inter_area
    return inter_area / union_area.clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Split-conformal calibration
# ---------------------------------------------------------------------------


def calibrate(scores: Sequence[float], delta: float) -> float:
    """
    Compute the split-conformal threshold τ^(t)_δ (§3.4, Eq. 15).

    Given calibration nonconformity scores {α_j} for a node type, select the
    ⌈(n+1)(1−δ)⌉-th order statistic as the threshold.

    Parameters
    ----------
    scores:
        Nonconformity scores α^(t)_j = s^(t)(x_j, z_j) for j=1…n.
    delta:
        Miscoverage level, δ ∈ (0, 1).  Coverage target is 1 − δ.

    Returns
    -------
    float
        Threshold τ^(t)_δ such that Γ^(t)_δ achieves ≥ (1−δ) coverage.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("calibrate() requires at least one calibration score.")
    sorted_scores = sorted(scores)
    k = math.ceil((n + 1) * (1 - delta))
    k = min(k, n)  # clamp to valid index
    return float(sorted_scores[k - 1])


def conformal_set(
    candidates: List[Any],
    candidate_scores: Sequence[float],
    threshold: float,
    k_max: Optional[int] = None,
) -> List[Any]:
    """
    Build the conformal prediction set Γ^(t)_δ(x_v) (§3.4, Eq. 16).

    Include candidate z if s^(t)(x_v, z) ≤ τ^(t)_δ.  Optionally cap the set
    at k_max elements (truncating to the least nonconforming candidates).

    Parameters
    ----------
    candidates:
        List of K candidate outputs.
    candidate_scores:
        Nonconformity score for each candidate, length K.
    threshold:
        τ^(t)_δ from ``calibrate``.
    k_max:
        Maximum allowed set size (Kmax in paper).

    Returns
    -------
    List of accepted candidates, sorted by ascending nonconformity score.
    """
    pairs = sorted(
        [(s, c) for s, c in zip(candidate_scores, candidates) if s <= threshold],
        key=lambda p: p[0],
    )
    if k_max is not None:
        pairs = pairs[:k_max]
    return [c for _, c in pairs]


# ---------------------------------------------------------------------------
# Trainable certificate head   s^(t)(x_v, z) = g^(t)_ψ(φ^(t)_θ(x_v, z))
# ---------------------------------------------------------------------------


class ConformityHead(nn.Module):
    """
    Lightweight MLP certificate head (§3.4, Eq. 18).

    Takes a feature vector φ^(t)_θ(x_v, z) – typically the hidden state of an
    MLLM or a vision backbone concatenated with candidate features – and
    outputs a scalar nonconformity score in [0, ∞).

    Parameters
    ----------
    input_dim:
        Dimension of the feature vector.
    hidden_dim:
        Width of the single hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),   # guarantees non-negative output
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : Tensor [..., input_dim]

        Returns
        -------
        Tensor [...] of non-negative nonconformity scores.
        """
        return self.net(features).squeeze(-1)


# ---------------------------------------------------------------------------
# Empirical coverage tracker
# ---------------------------------------------------------------------------


class CoverageTracker:
    """
    Running monitor for empirical coverage (§4.4, §4.6).

        Ĉov = (1/N) Σ 1[z_true ∈ Γ^(t)_δ(x_v)]

    Parameters
    ----------
    node_type:
        The node type being tracked.
    delta:
        Target miscoverage δ; coverage target is 1 − δ.
    """

    def __init__(self, node_type: NodeType, delta: float = 0.1) -> None:
        self.node_type = node_type
        self.delta = delta
        self._total = 0
        self._covered = 0

    def update(self, true_output: Any, pred_set: List[Any]) -> None:
        """Record whether true_output is in pred_set."""
        self._total += 1
        if true_output in pred_set:
            self._covered += 1

    @property
    def empirical_coverage(self) -> float:
        if self._total == 0:
            return float("nan")
        return self._covered / self._total

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.delta

    def __repr__(self) -> str:
        return (
            f"CoverageTracker({self.node_type.value}: "
            f"{self.empirical_coverage:.3f} / target {self.target_coverage:.3f}, "
            f"n={self._total})"
        )
