"""
Top-level PoP model class.

Provides a single ``ProofOfPerception`` entry-point that bundles:
- NodeRegistry          (perception + logic nodes)
- AdaptiveController    (budget-aware action policy)
- Calibrated thresholds  τ^(t)_δ per node type
- Split-CP calibration   API
- Training loss          PoPLoss
- Inference engine       PoP
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from pop.conformal import calibrate
from pop.controller import AdaptiveController
from pop.graph import NodeType
from pop.inference import PoP
from pop.nodes import NodeRegistry
from pop.training import PoPLoss


class ProofOfPerception(nn.Module):
    """
    Proof-of-Perception (PoP) model.

    This class aggregates all trainable and stateful components of PoP and
    exposes the main ``infer`` and ``calibrate_nodes`` APIs for evaluation,
    as well as ``compute_loss`` for training.

    Parameters
    ----------
    feature_dim:
        Embedding dimension shared by certificate heads and the controller.
    context_dim:
        Dimension of the contextual embedding input to the controller.
    budget:
        Per-sample compute budget B (§4.3, default 16).
    delta:
        Desired miscoverage level δ (default 0.1 → 90% coverage target).
    task_mode:
        ``'ce'`` or ``'smooth_l1'`` – task loss mode.
    gamma_plan, gamma_cert, gamma_ctrl:
        Loss combination weights (§3.7, Eq. 30).
    """

    def __init__(
        self,
        feature_dim: int = 256,
        context_dim: int = 256,
        budget: float = 16.0,
        delta: float = 0.1,
        task_mode: str = "ce",
        gamma_plan: float = 1.0,
        gamma_cert: float = 1.0,
        gamma_ctrl: float = 1.0,
    ) -> None:
        super().__init__()
        self.delta = delta
        self.budget = budget
        self.feature_dim = feature_dim
        self.context_dim = context_dim

        # ---- components ----
        self.registry = NodeRegistry.build(feature_dim=feature_dim)
        self.controller = AdaptiveController(
            context_dim=context_dim, hidden_dim=128
        )
        self.loss_fn = PoPLoss(
            task_mode=task_mode,
            gamma_plan=gamma_plan,
            gamma_cert=gamma_cert,
            gamma_ctrl=gamma_ctrl,
        )

        # Per-type calibrated thresholds (populated by calibrate_nodes)
        self.thresholds: Dict[str, float] = {nt.value: 1.0 for nt in NodeType}

        # Calibration score pools C^(t) (appended by self-play / calibration)
        self.calibration_pools: Dict[str, List[float]] = {
            nt.value: [] for nt in NodeType
        }

    # ------------------------------------------------------------------
    # Calibration (§3.4)
    # ------------------------------------------------------------------

    def add_calibration_score(self, node_type: NodeType, score: float) -> None:
        """Append a nonconformity score to pool C^(t)."""
        self.calibration_pools[node_type.value].append(score)

    def calibrate_nodes(self, delta: Optional[float] = None) -> Dict[str, float]:
        """
        Run split-CP calibration for each node type that has calibration scores.

        Updates ``self.thresholds`` and returns a dict of updated thresholds.

        Parameters
        ----------
        delta:
            Override the instance-level δ for this calibration run.
        """
        d = delta if delta is not None else self.delta
        updated: Dict[str, float] = {}
        for nt in NodeType:
            pool = self.calibration_pools[nt.value]
            if pool:
                tau = calibrate(pool, d)
                self.thresholds[nt.value] = tau
                updated[nt.value] = tau
        return updated

    # ------------------------------------------------------------------
    # Inference (§3.8)
    # ------------------------------------------------------------------

    def infer(
        self,
        images: List[Any],
        query: str,
        greedy: bool = True,
        planner: Optional[Any] = None,
    ) -> Tuple[List[Any], Any]:
        """
        Run two-phase PoP inference.

        Parameters
        ----------
        images:
            Input images (list of PIL / numpy / Tensor).
        query:
            Natural-language question.
        greedy:
            Use argmax controller action selection.
        planner:
            Optional callable(images, query) → DSL program string.

        Returns
        -------
        answer_set:
            Conformal set Γ^(answer)_δ(x) – the calibrated answer candidates.
        graph:
            The executed ReasoningGraph for inspection / logging.
        """
        engine = PoP(
            registry=self.registry,
            controller=self.controller,
            thresholds=self.thresholds,
            budget=self.budget,
            context_dim=self.context_dim,
            greedy=greedy,
            planner=planner,
        )
        return engine.run(images, query)

    # ------------------------------------------------------------------
    # Training (§3.7)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        task_logits: torch.Tensor,
        task_targets: torch.Tensor,
        plan_log_probs: torch.Tensor,
        plan_targets: torch.Tensor,
        cert_scores: torch.Tensor,
        cert_thresholds: torch.Tensor,
        cert_node_types: Optional[List[str]],
        ctrl_action_log_probs: torch.Tensor,
        ctrl_error_cost: torch.Tensor,
        ctrl_compute_cost: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the total training loss L (§3.7, Eq. 30).

        Returns
        -------
        total_loss : Tensor (scalar)
        components : Dict with keys 'task', 'plan', 'cert', 'ctrl'
        """
        return self.loss_fn(
            task_logits=task_logits,
            task_targets=task_targets,
            plan_log_probs=plan_log_probs,
            plan_targets=plan_targets,
            cert_scores=cert_scores,
            cert_thresholds=cert_thresholds,
            cert_node_types=cert_node_types,
            ctrl_action_log_probs=ctrl_action_log_probs,
            ctrl_error_cost=ctrl_error_cost,
            ctrl_compute_cost=ctrl_compute_cost,
        )

    def forward(self, images: List[Any], query: str) -> Tuple[List[Any], Any]:
        """Alias for ``infer`` to conform to nn.Module convention."""
        return self.infer(images, query)
