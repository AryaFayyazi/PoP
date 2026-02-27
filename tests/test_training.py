"""Tests for the training loss components (§3.7)."""

import pytest
import torch

from pop.training import CertificateLoss, PlanningLoss, PoPLoss, TaskLoss


class TestTaskLoss:
    def test_ce_mode(self):
        loss_fn = TaskLoss(mode="ce")
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert loss >= 0

    def test_smooth_l1_mode(self):
        loss_fn = TaskLoss(mode="smooth_l1")
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 1.9, 3.5])
        loss = loss_fn(preds, targets)
        assert loss >= 0

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            TaskLoss(mode="mse")

    def test_zero_loss_on_perfect_prediction(self):
        loss_fn = TaskLoss(mode="smooth_l1")
        vals = torch.tensor([1.0, 2.0, 3.0])
        loss = loss_fn(vals, vals)
        assert float(loss) == pytest.approx(0.0)


class TestPlanningLoss:
    def test_output_shape(self):
        loss_fn = PlanningLoss()
        B, T, V = 2, 5, 100
        log_probs = torch.log_softmax(torch.randn(B, T, V), dim=-1)
        targets = torch.randint(0, V, (B, T))
        loss = loss_fn(log_probs, targets)
        assert loss.shape == ()

    def test_ignore_index(self):
        loss_fn = PlanningLoss(ignore_index=-100)
        B, T, V = 2, 5, 50
        log_probs = torch.log_softmax(torch.randn(B, T, V), dim=-1)
        # All targets are the ignore index → should still return a tensor
        targets = torch.full((B, T), -100)
        loss = loss_fn(log_probs, targets)
        # NaN expected when all positions are masked, but tensor returned
        assert isinstance(loss, torch.Tensor)

    def test_lower_loss_on_higher_confidence(self):
        loss_fn = PlanningLoss()
        B, T, V = 1, 3, 10
        targets = torch.zeros(B, T, dtype=torch.long)  # always class 0
        # Confident on class 0
        conf_logits = torch.full((B, T, V), -10.0)
        conf_logits[:, :, 0] = 10.0
        conf_log_probs = torch.log_softmax(conf_logits, dim=-1)
        # Uniform
        unif_log_probs = torch.log_softmax(torch.zeros(B, T, V), dim=-1)
        loss_conf = float(loss_fn(conf_log_probs, targets))
        loss_unif = float(loss_fn(unif_log_probs, targets))
        assert loss_conf < loss_unif


class TestCertificateLoss:
    def test_zero_when_all_below_threshold(self):
        loss_fn = CertificateLoss(slack=0.0)
        scores = torch.tensor([0.1, 0.2, 0.3])
        thresholds = torch.tensor([0.5, 0.5, 0.5])
        loss = loss_fn(scores, thresholds)
        assert float(loss) == pytest.approx(0.0)

    def test_positive_when_above_threshold(self):
        loss_fn = CertificateLoss(slack=0.0)
        scores = torch.tensor([0.8, 0.9])
        thresholds = torch.tensor([0.5, 0.5])
        loss = loss_fn(scores, thresholds)
        assert float(loss) > 0.0
        assert float(loss) == pytest.approx(0.35)  # mean((0.3, 0.4))

    def test_slack_parameter(self):
        loss_fn = CertificateLoss(slack=0.1)
        scores = torch.tensor([0.55])
        thresholds = torch.tensor([0.5])
        # 0.55 − 0.5 − 0.1 = −0.05 → clamped to 0
        loss = loss_fn(scores, thresholds)
        assert float(loss) == pytest.approx(0.0)

    def test_type_weights(self):
        weights = {"ocr-string": 2.0, "det-box": 0.5}
        loss_fn = CertificateLoss(type_weights=weights)
        scores = torch.tensor([0.8, 0.8])
        thresholds = torch.tensor([0.5, 0.5])
        node_types = ["ocr-string", "det-box"]
        loss = loss_fn(scores, thresholds, node_types)
        expected = (2.0 * 0.3 + 0.5 * 0.3) / 2  # weighted mean
        assert float(loss) == pytest.approx(expected)


class TestPoPLoss:
    def _dummy_inputs(self):
        B, T, V = 2, 4, 20
        return dict(
            task_logits=torch.randn(B, V),
            task_targets=torch.randint(0, V, (B,)),
            plan_log_probs=torch.log_softmax(torch.randn(B, T, V), dim=-1),
            plan_targets=torch.randint(0, V, (B, T)),
            cert_scores=torch.tensor([0.2, 0.3, 0.6]),
            cert_thresholds=torch.tensor([0.5, 0.5, 0.5]),
            cert_node_types=None,
            ctrl_action_log_probs=torch.log(torch.tensor([0.4, 0.3, 0.3])),
            ctrl_error_cost=torch.tensor(0.0),
            ctrl_compute_cost=torch.tensor(5.0),
        )

    def test_total_loss_positive(self):
        loss_fn = PoPLoss()
        inputs = self._dummy_inputs()
        total, components = loss_fn(**inputs)
        assert total > 0
        assert set(components.keys()) == {"task", "plan", "cert", "ctrl"}

    def test_loss_weights_applied(self):
        # gamma_plan=0 should make plan contribution vanish
        loss_fn_full = PoPLoss(gamma_plan=1.0)
        loss_fn_no_plan = PoPLoss(gamma_plan=0.0)
        inputs = self._dummy_inputs()
        total_full, _ = loss_fn_full(**inputs)
        total_no_plan, comps = loss_fn_no_plan(**inputs)
        # Difference should roughly equal the plan component
        diff = float(total_full - total_no_plan)
        plan_val = float(comps["plan"])
        assert diff == pytest.approx(plan_val, rel=1e-4)

    def test_returns_components_dict(self):
        loss_fn = PoPLoss()
        inputs = self._dummy_inputs()
        total, comps = loss_fn(**inputs)
        for key in ("task", "plan", "cert", "ctrl"):
            assert isinstance(comps[key], torch.Tensor)
            assert comps[key].shape == ()
