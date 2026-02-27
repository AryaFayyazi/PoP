"""
Node implementations for Proof-of-Perception (PoP) (§3.2–§3.4).

Each node type corresponds to a perception or logic-fusion operation in the
reasoning DAG G = (V, E).  All nodes share a common ``execute`` interface that:

    1. Generates a set of candidate outputs Z^cand_v.
    2. Computes nonconformity scores s^(t)(x_v, z) for every candidate.
    3. Filters candidates with the calibrated conformal threshold τ^(t)_δ.
    4. Returns the conformal prediction set Γ^(t)_δ(x_v).

Supported node types
---------------------
- ``OCRNode``          – string-valued OCR (Eq. 9)
- ``DetectionNode``    – bounding-box detection (Eq. 10)
- ``ChartNode``        – numeric chart parsing (Eq. 11)
- ``LogicFusionNode``  – MLLM reasoning fusion with trainable ConformityHead (Eq. 18)

``NodeRegistry`` maps tool-key strings to ``BaseNode`` instances and is the
sole factory used by the inference engine.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from pop.conformal import (
    ConformityHead,
    conformal_set,
    nonconformity_box,
    nonconformity_numeric,
    nonconformity_ocr,
)
from pop.graph import NodeSpec, NodeType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseNode(abc.ABC):
    """
    Abstract base class for all PoP reasoning-node implementations (§3.2).

    Concrete subclasses must implement ``execute``, which runs the node,
    scores candidates with the appropriate nonconformity function, and
    returns the conformal prediction set Γ^(t)_δ(x_v).
    """

    @abc.abstractmethod
    def execute(
        self,
        spec: NodeSpec,
        images: List[Any],
        threshold: float,
        context: Dict[str, Any],
    ) -> List[Any]:
        """
        Execute the node and return the conformal prediction set.

        Parameters
        ----------
        spec:
            NodeSpec carrying tool_key, region, prompt, is_fusion, parents.
        images:
            Full list of input images (PIL / numpy / Tensor).
        threshold:
            Pre-computed τ^(t)_δ for this node type.
        context:
            Dict mapping parent node_id → its conformal set, plus
            ``'__query__'`` → natural-language query string.

        Returns
        -------
        List of accepted candidate outputs (the conformal set Γ^(t)_δ(x_v)).
        """


# ---------------------------------------------------------------------------
# OCR node  (§3.3, Eq. 9)
# ---------------------------------------------------------------------------


class OCRNode(BaseNode):
    """
    String-valued OCR perception node.

    Nonconformity score (Eq. 9):
        s^(ocr)(x_v, z) = 1 − P_θ(z | x_v)

    In production, replace the stub candidate generation in ``execute`` with
    a real OCR / MLLM beam-search call that returns ``(strings, probs)``.
    The ConformityHead (Eq. 18) is available as ``self.cert_head`` for use
    when a learned rather than analytical scoring is preferred.

    Parameters
    ----------
    feature_dim:
        Dimension of the certificate-head input φ^(t)_θ(x_v, z).
    """

    #: Maximum conformal-set size K_max (§4.3)
    K_MAX: int = 5

    def __init__(self, feature_dim: int = 256) -> None:
        self.feature_dim = feature_dim
        # Trainable certificate head  g^(t)_ψ  (Eq. 18)
        self.cert_head = ConformityHead(input_dim=feature_dim)

    def execute(
        self,
        spec: NodeSpec,
        images: List[Any],
        threshold: float,
        context: Dict[str, Any],
    ) -> List[Any]:
        """
        Return the conformal set Γ^(ocr)_δ(x_v) of OCR string candidates.

        Stub implementation produces synthetic candidates with simulated
        model probabilities.  In production, call the OCR tool / MLLM and
        supply real ``(candidates, probs)``; the conformal filtering logic
        below remains unchanged.
        """
        # ---- Candidate generation (stub; replace with real OCR call) ----
        # In production:
        #   strings, probs = ocr_tool(image_crop, prompt=spec.prompt)
        candidates: List[str] = [
            "candidate_0",
            "candidate_1",
            "candidate_2",
            "candidate_3",
            "candidate_4",
        ]
        # Simulated token probabilities (sum ≤ 1; first = MAP prediction)
        model_probs = torch.tensor([0.60, 0.20, 0.10, 0.07, 0.03])
        candidate_ids = torch.arange(len(candidates))

        # ---- Nonconformity scores  s^(ocr)(x_v, z) = 1 − P_θ(z | x_v) ----
        scores = nonconformity_ocr(model_probs, candidate_ids)  # [K] ∈ [0, 1]

        # ---- Conformal filtering (Eq. 16) --------------------------------
        result = conformal_set(
            candidates, scores.tolist(), threshold, k_max=self.K_MAX
        )
        # Safety fallback: always return at least the MAP prediction
        if not result:
            result = [candidates[0]]

        logger.debug(
            "OCRNode '%s': |Γ|=%d, threshold=%.3f",
            spec.node_id, len(result), threshold,
        )
        return result


# ---------------------------------------------------------------------------
# Detection node  (§3.3, Eq. 10)
# ---------------------------------------------------------------------------


class DetectionNode(BaseNode):
    """
    Bounding-box object-detection perception node.

    Nonconformity score (Eq. 10):
        s^(box)(x_v, z) = 1 − IoU(z, ẑ_MAP)

    Candidates are bounding boxes [x1, y1, x2, y2].  The MAP box (highest
    detector confidence) anchors the IoU-based nonconformity computation.

    Parameters
    ----------
    feature_dim:
        Dimension of the certificate-head input.
    """

    #: Maximum conformal-set size K_max (§4.3)
    K_MAX: int = 3

    def __init__(self, feature_dim: int = 256) -> None:
        self.feature_dim = feature_dim
        self.cert_head = ConformityHead(input_dim=feature_dim)

    def execute(
        self,
        spec: NodeSpec,
        images: List[Any],
        threshold: float,
        context: Dict[str, Any],
    ) -> List[Any]:
        """
        Return the conformal set Γ^(det)_δ(x_v) of bounding-box candidates.

        Stub implementation; replace with a real object-detector call that
        returns ``(boxes_tensor, scores_tensor)`` and update the MAP box
        accordingly.
        """
        # ---- Candidate generation (stub; replace with real detector) -----
        # In production:
        #   boxes, det_scores = detector(image_crop, prompt=spec.prompt)
        #   map_box = boxes[det_scores.argmax()]
        candidate_boxes_data: List[List[float]] = [
            [10.0, 10.0, 100.0, 100.0],   # MAP box (highest detector score)
            [12.0, 12.0, 102.0,  98.0],   # slight positive shift
            [ 8.0,  8.0,  98.0, 102.0],   # slight negative shift
        ]
        candidate_boxes = torch.tensor(candidate_boxes_data)  # [K, 4]
        map_box = candidate_boxes[0]                           # [4]

        # ---- Nonconformity scores  s^(box)(x_v, z) = 1 − IoU(z, ẑ_MAP) ---
        scores = nonconformity_box(map_box, candidate_boxes)   # [K] ∈ [0, 1]

        # Convert to plain tuple list for serializability
        result_boxes = [tuple(b) for b in candidate_boxes_data]

        # ---- Conformal filtering (Eq. 16) ---------------------------------
        result = conformal_set(
            result_boxes, scores.tolist(), threshold, k_max=self.K_MAX
        )
        if not result:
            result = [tuple(candidate_boxes_data[0])]

        logger.debug(
            "DetectionNode '%s': |Γ|=%d, threshold=%.3f",
            spec.node_id, len(result), threshold,
        )
        return result


# ---------------------------------------------------------------------------
# Chart node  (§3.3, Eq. 11)
# ---------------------------------------------------------------------------


class ChartNode(BaseNode):
    """
    Numeric chart-parsing perception node.

    Nonconformity score (Eq. 11):
        s^(num)(x_v, z) = |z − μ_θ(x_v)|

    Produces a conformal interval around the predicted mean value μ_θ(x_v),
    discretized into K candidate numeric values.  For chart-num nodes, §4.3
    notes that the conformal set corresponds to a numeric interval.

    Parameters
    ----------
    feature_dim:
        Dimension of the certificate-head input.
    """

    K_MAX: int = 5

    def __init__(self, feature_dim: int = 256) -> None:
        self.feature_dim = feature_dim
        self.cert_head = ConformityHead(input_dim=feature_dim)

    def execute(
        self,
        spec: NodeSpec,
        images: List[Any],
        threshold: float,
        context: Dict[str, Any],
    ) -> List[Any]:
        """
        Return the conformal set Γ^(chart)_δ(x_v) of numeric chart values.

        Stub implementation; replace with a real chart-parser / MLLM numeric
        prediction that supplies ``pred_mean`` and ``candidates_raw``.
        """
        # ---- Candidate generation (stub; replace with real chart parser) -
        # In production:
        #   pred_mean, candidates_raw = chart_parser(image_crop, prompt=spec.prompt)
        pred_mean = torch.tensor(42.0)
        candidates_raw: List[float] = [42.0, 41.0, 43.0, 40.0, 44.0]
        candidates_tensor = torch.tensor(candidates_raw)

        # ---- Nonconformity scores  s^(num)(x_v, z) = |z − μ_θ(x_v)| -----
        scores = nonconformity_numeric(pred_mean, candidates_tensor)  # [K] ≥ 0

        # ---- Conformal filtering (Eq. 16) ---------------------------------
        result = conformal_set(
            candidates_raw, scores.tolist(), threshold, k_max=self.K_MAX
        )
        if not result:
            result = [candidates_raw[0]]

        logger.debug(
            "ChartNode '%s': |Γ|=%d, threshold=%.3f",
            spec.node_id, len(result), threshold,
        )
        return result


# ---------------------------------------------------------------------------
# Logic-fusion node  (§3.2 Eq. 6; §3.4 Eq. 18)
# ---------------------------------------------------------------------------


class LogicFusionNode(BaseNode):
    """
    MLLM logic-fusion node that aggregates upstream conformal sets.

    Implements the fusion step (Eq. 6):
        Z_v = f^(fuse)_θ(q, x, {Ẑ_u}_{u ∈ pa(v)})

    Uses a trainable ConformityHead (Eq. 18) to score candidate text answers:
        s^(logic)(x_v, z) = g^(logic)_ψ(φ^(logic)_θ(x_v, z))

    The certificate head takes a feature vector of dimension ``feature_dim``
    produced by the MLLM hidden state (or a placeholder projection in the
    stub) and outputs a scalar nonconformity score in [0, ∞).

    Parameters
    ----------
    feature_dim:
        Dimension of the MLLM hidden state / certificate-head input.
    """

    K_MAX: int = 5

    def __init__(self, feature_dim: int = 256) -> None:
        self.feature_dim = feature_dim
        # Trainable certificate head  g^(logic)_ψ  (Eq. 18)
        self.cert_head = ConformityHead(input_dim=feature_dim)
        # Linear projection simulating the MLLM hidden-state extraction step.
        # In production, replace with the actual MLLM hidden-state encoder.
        self.input_proj = nn.Linear(feature_dim, feature_dim)

    def execute(
        self,
        spec: NodeSpec,
        images: List[Any],
        threshold: float,
        context: Dict[str, Any],
    ) -> List[Any]:
        """
        Return the conformal set Γ^(logic)_δ(x_v) of fused text candidates.

        Assembles a structured prompt from the query and upstream conformal
        sets, generates candidate answers via MLLM (stub), scores each with
        the ConformityHead, and filters by the calibrated threshold.

        In production, replace the stub candidate generation with a real
        MLLM beam-search / sampling step that returns
        ``(candidate_texts, hidden_states)``.
        """
        query = context.get("__query__", "")

        # Collect upstream conformal sets from parent nodes
        parent_sets: List[List[Any]] = [
            context.get(pid, []) for pid in (spec.parents or [])
        ]

        # ---- Candidate generation (stub; replace with MLLM forward pass) -
        # In production:
        #   prompt = build_fusion_prompt(query, parent_sets, spec.prompt)
        #   candidate_texts, hidden_states = mllm.beam_search(prompt, k=K_MAX)
        candidates: List[str] = [
            "answer_A",
            "answer_B",
            "answer_C",
            "answer_D",
            "answer_E",
        ]
        # Simulated MLLM hidden states: [K, feature_dim]
        features = torch.randn(len(candidates), self.feature_dim)
        features = self.input_proj(features)

        # ---- Nonconformity scores via ConformityHead (Eq. 18) ------------
        with torch.no_grad():
            scores: List[float] = self.cert_head(features).tolist()

        # ---- Conformal filtering (Eq. 16) --------------------------------
        result = conformal_set(
            candidates, scores, threshold, k_max=self.K_MAX
        )
        # Fallback: return the least nonconforming candidate
        if not result:
            best_idx = int(torch.tensor(scores).argmin().item())
            result = [candidates[best_idx]]

        logger.debug(
            "LogicFusionNode '%s': |Γ|=%d, threshold=%.3f, parents=%s",
            spec.node_id, len(result), threshold, spec.parents,
        )
        return result


# ---------------------------------------------------------------------------
# Node registry
# ---------------------------------------------------------------------------

# Default mapping from tool-key to node class
_TOOL_KEY_TO_CLASS: Dict[str, type] = {
    "ocr":      OCRNode,
    "det":      DetectionNode,
    "detector": DetectionNode,
    "chart":    ChartNode,
    "layout":   LogicFusionNode,
    "vqa":      LogicFusionNode,
    "fuse":     LogicFusionNode,
}


class NodeRegistry:
    """
    Registry mapping tool-key strings to ``BaseNode`` instances (§3.2).

    A single ``NodeRegistry`` is created per ``ProofOfPerception`` model via
    the ``build`` class method.  All tool calls share the same node instances
    so certificate-head weights are updated consistently during training.

    Parameters
    ----------
    nodes:
        Mapping from tool_key (str) to BaseNode instance.
    """

    def __init__(self, nodes: Dict[str, BaseNode]) -> None:
        self._nodes = nodes

    @classmethod
    def build(cls, feature_dim: int = 256) -> "NodeRegistry":
        """
        Instantiate one node per concrete class and return a populated registry.

        Shared instances mean that certificate heads receive gradients from
        all tool calls of the same type, enabling joint training.

        Parameters
        ----------
        feature_dim:
            Embedding dimension passed to every node's ConformityHead.
        """
        ocr_node   = OCRNode(feature_dim=feature_dim)
        det_node   = DetectionNode(feature_dim=feature_dim)
        chart_node = ChartNode(feature_dim=feature_dim)
        fuse_node  = LogicFusionNode(feature_dim=feature_dim)

        _class_to_instance = {
            OCRNode:          ocr_node,
            DetectionNode:    det_node,
            ChartNode:        chart_node,
            LogicFusionNode:  fuse_node,
        }

        nodes: Dict[str, BaseNode] = {
            key: _class_to_instance[cls_]
            for key, cls_ in _TOOL_KEY_TO_CLASS.items()
        }
        return cls(nodes)

    def get(self, tool_key: str) -> Optional[BaseNode]:
        """Return the BaseNode for *tool_key*, or ``None`` if not registered."""
        return self._nodes.get(tool_key)

    def register(self, tool_key: str, node: BaseNode) -> None:
        """Register a custom node implementation at runtime."""
        self._nodes[tool_key] = node

    def __repr__(self) -> str:
        return f"NodeRegistry(keys={sorted(self._nodes.keys())})"


__all__ = [
    "BaseNode",
    "OCRNode",
    "DetectionNode",
    "ChartNode",
    "LogicFusionNode",
    "NodeRegistry",
]
