"""
Training objectives for PoP (§3.7).

The total training loss combines four terms (Eq. 30):

    L = L_task + γ_plan · L_plan + γ_cert · L_cert + γ_ctrl · L_ctrl

- L_task   – cross-entropy / smooth-L1 over final answers (Eq. 22)
- L_plan   – sequence-level negative log-likelihood for planner programs (Eq. 23)
- L_cert   – margin-based certificate loss aligning s^(t)(x_v, z_v) with τ^(t)_δ (Eq. 24–25)
- L_ctrl   – policy-gradient controller loss (Eq. 28–29); provided in controller.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Task loss   L_task   (Eq. 22)
# ---------------------------------------------------------------------------


class TaskLoss(nn.Module):
    """
    Task loss for the final answer prediction.

    Supports two modes:
    - ``'ce'``      – cross-entropy for classification (e.g. MCQ, categorical)
    - ``'smooth_l1'`` – smooth L1 for regression / numeric QA

    Parameters
    ----------
    mode:
        ``'ce'`` or ``'smooth_l1'``.
    """

    def __init__(self, mode: str = "ce") -> None:
        super().__init__()
        if mode not in ("ce", "smooth_l1"):
            raise ValueError(f"Unknown mode '{mode}'. Choose 'ce' or 'smooth_l1'.")
        self.mode = mode

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : Tensor [B, C] (classification) or [B] (regression)
        targets : Tensor [B]    (class indices)   or [B] (target values)
        """
        if self.mode == "ce":
            return F.cross_entropy(logits, targets)
        else:
            return F.smooth_l1_loss(logits, targets.float())


# ---------------------------------------------------------------------------
# Planning loss   L_plan   (Eq. 23)
# ---------------------------------------------------------------------------


class PlanningLoss(nn.Module):
    """
    Sequence-level negative log-likelihood loss for the planner program.

        L_plan = E_{(x, π⋆)} [ −log p_θ(π⋆ | x) ]

    Expects token-level log-probabilities from the MLLM decoder.

    Parameters
    ----------
    ignore_index:
        Token ID to ignore in the NLL (e.g. padding, default −100).
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        log_probs: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        log_probs    : Tensor [B, T, V]  – per-token log-probabilities from decoder.
        target_tokens: Tensor [B, T]     – ground-truth DSL token ids.

        Returns
        -------
        Scalar loss (mean over valid tokens in the batch).
        """
        B, T, V = log_probs.shape
        return F.nll_loss(
            log_probs.reshape(B * T, V),
            target_tokens.reshape(B * T),
            ignore_index=self.ignore_index,
        )


# ---------------------------------------------------------------------------
# Certificate loss   L_cert   (Eq. 24–25)
# ---------------------------------------------------------------------------


class CertificateLoss(nn.Module):
    """
    Margin-based loss ensuring s^(t)(x_v, z_v) ≤ τ^(t)_δ + ε for true outputs.

        ℓ^(t)_cert(x_v, z_v) = max(0, s^(t)(x_v, z_v) − τ^(t)_δ − ε)

    Aggregated over node types with per-type weights λ_t (Eq. 25).

    Parameters
    ----------
    type_weights:
        Dict mapping NodeType.value → λ_t weight.  Missing types default to 1.
    slack:
        ε ≥ 0 – slack margin that encourages true outputs to score below τ.
    """

    def __init__(
        self,
        type_weights: Optional[Dict[str, float]] = None,
        slack: float = 0.0,
    ) -> None:
        super().__init__()
        self.type_weights = type_weights or {}
        self.slack = slack

    def forward(
        self,
        scores: torch.Tensor,
        thresholds: torch.Tensor,
        node_type_keys: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        scores     : Tensor [N]  – nonconformity scores s^(t)(x_v, z_v) for true z_v.
        thresholds : Tensor [N]  – per-node calibrated thresholds τ^(t)_δ.
        node_type_keys : list of N NodeType.value strings (for per-type weights).

        Returns
        -------
        Scalar loss.
        """
        margins = F.relu(scores - thresholds - self.slack)   # Eq. 24
        if node_type_keys:
            weights = torch.tensor(
                [self.type_weights.get(k, 1.0) for k in node_type_keys],
                dtype=scores.dtype,
                device=scores.device,
            )
            loss = (margins * weights).mean()
        else:
            loss = margins.mean()
        return loss


# ---------------------------------------------------------------------------
# Combined objective   L   (Eq. 30)
# ---------------------------------------------------------------------------


class PoPLoss(nn.Module):
    """
    Total PoP training loss (Eq. 30):

        L = L_task + γ_plan · L_plan + γ_cert · L_cert + γ_ctrl · L_ctrl

    Parameters
    ----------
    task_mode:
        ``'ce'`` or ``'smooth_l1'`` – passed to TaskLoss.
    gamma_plan, gamma_cert, gamma_ctrl:
        Loss combination weights (default 1.0 each).
    cert_type_weights:
        Per-node-type weights λ_t for the certificate loss.
    cert_slack:
        Margin slack ε for the certificate loss.
    ctrl_beta:
        Compute penalty β for the controller cost C(x) = C_err + β C_comp.
    """

    def __init__(
        self,
        task_mode: str = "ce",
        gamma_plan: float = 1.0,
        gamma_cert: float = 1.0,
        gamma_ctrl: float = 1.0,
        cert_type_weights: Optional[Dict[str, float]] = None,
        cert_slack: float = 0.0,
        ctrl_beta: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma_plan = gamma_plan
        self.gamma_cert = gamma_cert
        self.gamma_ctrl = gamma_ctrl
        self.task_loss = TaskLoss(mode=task_mode)
        self.plan_loss = PlanningLoss()
        self.cert_loss = CertificateLoss(
            type_weights=cert_type_weights, slack=cert_slack
        )
        from pop.controller import ControllerLoss
        self.ctrl_loss = ControllerLoss(beta=ctrl_beta)

    def forward(
        self,
        # Task
        task_logits: torch.Tensor,
        task_targets: torch.Tensor,
        # Plan
        plan_log_probs: torch.Tensor,
        plan_targets: torch.Tensor,
        # Certificate
        cert_scores: torch.Tensor,
        cert_thresholds: torch.Tensor,
        cert_node_types: Optional[List[str]],
        # Controller
        ctrl_action_log_probs: torch.Tensor,
        ctrl_error_cost: torch.Tensor,
        ctrl_compute_cost: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the total loss and return a dict of individual components for
        logging.

        Returns
        -------
        total_loss : Tensor (scalar)
        components : Dict with keys 'task', 'plan', 'cert', 'ctrl'
        """
        l_task = self.task_loss(task_logits, task_targets)
        l_plan = self.plan_loss(plan_log_probs, plan_targets)
        l_cert = self.cert_loss(cert_scores, cert_thresholds, cert_node_types)
        l_ctrl = self.ctrl_loss(
            ctrl_action_log_probs, ctrl_error_cost, ctrl_compute_cost
        )
        total = (
            l_task
            + self.gamma_plan * l_plan
            + self.gamma_cert * l_cert
            + self.gamma_ctrl * l_ctrl
        )
        return total, {
            "task": l_task,
            "plan": l_plan,
            "cert": l_cert,
            "ctrl": l_ctrl,
        }
