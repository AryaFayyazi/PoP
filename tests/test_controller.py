"""Tests for the AdaptiveController and BudgetTracker (§3.5)."""

import pytest
import torch
import torch.nn.functional as F

from pop.controller import (
    AdaptiveController,
    BudgetTracker,
    ControllerLoss,
    encode_certificate_state,
    _CERT_STATE_DIM,
)
from pop.graph import Action, NodeSpec, NodeType


# ---------------------------------------------------------------------------
# encode_certificate_state
# ---------------------------------------------------------------------------


class TestEncodeCertificateState:
    def test_output_dim(self):
        spec = NodeSpec(node_id="v0", node_type=NodeType.OCR)
        vec = encode_certificate_state(spec, threshold=0.5, set_size=3)
        assert vec.shape == (_CERT_STATE_DIM,)  # 2 + len(NodeType) = 6

    def test_threshold_and_set_size_encoded(self):
        spec = NodeSpec(node_id="v0", node_type=NodeType.DETECTION)
        vec = encode_certificate_state(spec, threshold=0.7, set_size=2)
        assert float(vec[0]) == pytest.approx(0.7)
        assert float(vec[1]) == pytest.approx(2.0)

    def test_one_hot_sums_to_one(self):
        for nt in NodeType:
            spec = NodeSpec(node_id="v", node_type=nt)
            vec = encode_certificate_state(spec, threshold=0.5, set_size=1)
            one_hot = vec[2:]
            assert float(one_hot.sum()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AdaptiveController
# ---------------------------------------------------------------------------


CONTEXT_DIM = 64


class TestAdaptiveController:
    def _make_controller(self):
        return AdaptiveController(context_dim=CONTEXT_DIM, hidden_dim=32)

    def test_forward_output_shape(self):
        ctrl = self._make_controller()
        B = 4
        cert = torch.randn(B, _CERT_STATE_DIM)
        budget = torch.rand(B, 1)
        ctx = torch.randn(B, CONTEXT_DIM)
        logits = ctrl(cert, budget, ctx)
        assert logits.shape == (B, 4)

    def test_select_action_returns_valid_action(self):
        ctrl = self._make_controller()
        cert = torch.randn(_CERT_STATE_DIM)
        budget = torch.rand(1)
        ctx = torch.randn(CONTEXT_DIM)
        action = ctrl.select_action(cert, budget, ctx)
        assert action in list(Action)

    def test_greedy_deterministic(self):
        ctrl = self._make_controller()
        cert = torch.randn(_CERT_STATE_DIM)
        budget = torch.rand(1)
        ctx = torch.randn(CONTEXT_DIM)
        actions = {ctrl.select_action(cert, budget, ctx, greedy=True) for _ in range(5)}
        assert len(actions) == 1  # always same action

    def test_action_probabilities_sum_to_one(self):
        ctrl = self._make_controller()
        cert = torch.randn(_CERT_STATE_DIM)
        budget = torch.rand(1)
        ctx = torch.randn(CONTEXT_DIM)
        logits = ctrl(cert.unsqueeze(0), budget.unsqueeze(0), ctx.unsqueeze(0))
        probs = F.softmax(logits.squeeze(0), dim=-1)
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------


class TestBudgetTracker:
    def test_initial_state(self):
        bt = BudgetTracker(total_budget=16.0)
        assert bt.remaining == pytest.approx(16.0)
        assert not bt.exhausted()

    def test_spend(self):
        bt = BudgetTracker(total_budget=10.0)
        bt.spend(3.0)
        assert bt.remaining == pytest.approx(7.0)

    def test_exhausted(self):
        bt = BudgetTracker(total_budget=5.0)
        bt.spend(5.0)
        assert bt.exhausted()

    def test_remaining_clamped_at_zero(self):
        bt = BudgetTracker(total_budget=5.0)
        bt.spend(10.0)
        assert bt.remaining == pytest.approx(0.0)

    def test_fraction_remaining(self):
        bt = BudgetTracker(total_budget=10.0)
        bt.spend(4.0)
        assert bt.fraction_remaining == pytest.approx(0.6)

    def test_as_tensor(self):
        bt = BudgetTracker(total_budget=8.0)
        bt.spend(2.0)
        t = bt.as_tensor()
        assert t.shape == (1,)
        assert float(t[0]) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# ControllerLoss
# ---------------------------------------------------------------------------


class TestControllerLoss:
    def test_loss_is_scalar(self):
        loss_fn = ControllerLoss(beta=0.05)
        log_probs = torch.log(torch.tensor([0.5, 0.3, 0.2]))
        err = torch.tensor(1.0)
        comp = torch.tensor(10.0)
        loss = loss_fn(log_probs, err, comp)
        assert loss.shape == ()

    def test_baseline_updated(self):
        loss_fn = ControllerLoss(beta=0.05, ema_alpha=0.9)
        log_probs = torch.log(torch.tensor([0.5]))
        for _ in range(5):
            loss_fn(log_probs, torch.tensor(0.0), torch.tensor(5.0))
        # baseline should have moved away from 0
        assert loss_fn._baseline != 0.0

    def test_lower_error_lower_loss_tendency(self):
        """
        Higher reward (lower cost) → advantage > negative → loss is less negative.
        With low_err: R=0, advantage=0 → loss=0.
        With high_err: R=-1, advantage=-1 → loss=-(-0.5*-1)=-0.5 (more negative).
        So l_low > l_high.
        """
        loss_fn_low = ControllerLoss(beta=0.0)
        loss_fn_high = ControllerLoss(beta=0.0)
        log_probs = torch.tensor([-0.5, -0.5])
        low_err = torch.tensor(0.0)
        high_err = torch.tensor(1.0)
        comp = torch.tensor(0.0)
        # Reset baselines to 0
        loss_fn_low._baseline = 0.0
        loss_fn_high._baseline = 0.0
        l_low = loss_fn_low(log_probs, low_err, comp)
        l_high = loss_fn_high(log_probs, high_err, comp)
        # Lower error → higher reward → less penalising loss value
        assert float(l_low) > float(l_high)
