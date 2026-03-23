"""
Adaptive Controller for PoP (§3.5).

The controller πϕ observes per-node certificate state c_v, the remaining
compute budget b, and contextual information to choose an action from
    A = {ACCEPT, RETRY, EXPAND, ABORT}.

Architecture: a small MLP policy that maps (c_v, b, context_embedding) → logits
over A.  At training time, the controller is optimised with REINFORCE (policy
gradients) using reward R = −C_err − β C_comp (§3.7, Eq. 26–29).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pop.graph import Action, NodeSpec, NodeType


# ---------------------------------------------------------------------------
# Certificate state encoding (Eq. 19)
# ---------------------------------------------------------------------------

# One-hot dimension for NodeType
_NODE_TYPE_DIM = len(NodeType)

# Fixed certificate-state feature size:
#   1 (threshold τ) + 1 (set size |Γ|) + _NODE_TYPE_DIM (type one-hot)
_CERT_STATE_DIM = 2 + _NODE_TYPE_DIM


def encode_certificate_state(
    spec: NodeSpec,
    threshold: float,
    set_size: int,
) -> torch.Tensor:
    """
    Encode c_v = (τ^(t)_δ, |Γ^(t)_δ(x_v)|, type(v)) as a flat float tensor.

    The ground-truth coverage indicator 1[z_true ∈ Γ] is only available at
    training time and is handled separately via the controller loss (§3.7).

    Returns
    -------
    Tensor [_CERT_STATE_DIM]
    """
    type_list = list(NodeType)
    one_hot = torch.zeros(_NODE_TYPE_DIM)
    try:
        one_hot[type_list.index(spec.node_type)] = 1.0
    except ValueError:
        pass  # unknown type → zero vector
    return torch.cat(
        [torch.tensor([threshold, float(set_size)], dtype=torch.float32), one_hot]
    )


# ---------------------------------------------------------------------------
# Controller network
# ---------------------------------------------------------------------------


class AdaptiveController(nn.Module):
    """
    Policy network πϕ that selects actions for each reasoning node (§3.5).

    Parameters
    ----------
    context_dim:
        Dimension of the contextual embedding (query + upstream summary).
        Defaults to 256.
    hidden_dim:
        Hidden layer width.

    Input to forward():
        cert_state  [B, _CERT_STATE_DIM]   – certificate state c_v
        budget      [B, 1]                 – remaining normalised budget
        context     [B, context_dim]       – query / upstream embedding
    Output:
        logits [B, 4] over {ACCEPT, RETRY, EXPAND, ABORT} (in that order).
    """

    _ACTIONS: List[Action] = [
        Action.ACCEPT,
        Action.RETRY,
        Action.EXPAND,
        Action.ABORT,
    ]

    def __init__(self, context_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        input_dim = _CERT_STATE_DIM + 1 + context_dim  # cert + budget + ctx
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self._ACTIONS)),
        )

    def forward(
        self,
        cert_state: torch.Tensor,
        budget: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Return action logits."""
        x = torch.cat([cert_state, budget, context], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def select_action(
        self,
        cert_state: torch.Tensor,
        budget: torch.Tensor,
        context: torch.Tensor,
        greedy: bool = False,
    ) -> Action:
        """
        Sample (or greedily select) an action for a single node.

        Parameters
        ----------
        cert_state : Tensor [_CERT_STATE_DIM]
        budget     : Tensor [1]
        context    : Tensor [context_dim]
        greedy     : If True use argmax; otherwise sample from the distribution.

        Returns
        -------
        Action
        """
        logits = self.forward(
            cert_state.unsqueeze(0),
            budget.unsqueeze(0),
            context.unsqueeze(0),
        ).squeeze(0)  # [4]
        if greedy:
            idx = int(logits.argmax().item())
        else:
            idx = int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())
        return self._ACTIONS[idx]


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------


class BudgetTracker:
    """
    Tracks cumulative computation cost during graph execution (§3.5, §4.3).

    Default costs match the paper:
        tool call    → 1 unit
        high-res retry → 2 units
        fusion step  → 0.25 units
    """

    TOOL_COST: float = 1.0
    RETRY_COST: float = 2.0
    FUSE_COST: float = 0.25

    def __init__(self, total_budget: float = 16.0) -> None:
        self.total_budget = total_budget
        self._spent: float = 0.0

    def spend(self, amount: float) -> None:
        self._spent += amount

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_budget - self._spent)

    @property
    def fraction_remaining(self) -> float:
        return self.remaining / self.total_budget

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor([self.fraction_remaining], dtype=torch.float32)

    def exhausted(self) -> bool:
        return self._spent >= self.total_budget


# ---------------------------------------------------------------------------
# REINFORCE loss for the controller (§3.7, Eq. 28–29)
# ---------------------------------------------------------------------------


class ControllerLoss(nn.Module):
    """
    Policy-gradient (REINFORCE) loss for the adaptive controller.

        L_ctrl = −E[ Σ_v log πϕ(a_v | c_v, b, ctx(v)) · (R(x) − b0) ]

    where R(x) = −C(x) = −(C_err + β · C_comp)  (Eq. 26).

    Parameters
    ----------
    beta:
        Compute penalty weight β (default 0.05, matching §4.3).
    ema_alpha:
        Exponential moving-average coefficient for the baseline b0.
    """

    def __init__(self, beta: float = 0.05, ema_alpha: float = 0.99) -> None:
        super().__init__()
        self.beta = beta
        self.ema_alpha = ema_alpha
        self._baseline: float = 0.0

    def forward(
        self,
        action_log_probs: torch.Tensor,
        error_cost: torch.Tensor,
        compute_cost: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        action_log_probs : Tensor [T]
            Log-prob log πϕ(a_v | …) for each of the T node actions taken.
        error_cost : Tensor [] (scalar)
            C_err(x) – 0 if correct, 1 if wrong or coverage violated.
        compute_cost : Tensor [] (scalar)
            C_comp(x) – total tool / fusion cost for the example.

        Returns
        -------
        Scalar loss tensor.
        """
        reward = -(error_cost + self.beta * compute_cost)
        advantage = reward - self._baseline
        # Update EMA baseline (detached)
        with torch.no_grad():
            self._baseline = (
                self.ema_alpha * self._baseline
                + (1 - self.ema_alpha) * reward.item()
            )
        # REINFORCE gradient estimator: − log π · advantage
        loss = -(action_log_probs * advantage.detach()).mean()
        return loss
