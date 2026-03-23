"""Tests for the self-play counterexample mining module (§3.6)."""

import pytest

from pop.self_play import (
    PerturbationType,
    SelfPlayMiner,
    apply_perturbation,
)


class TestApplyPerturbation:
    def test_none_perturbation_identity(self):
        images = ["img1", "img2"]
        result = apply_perturbation(images, PerturbationType.NONE)
        assert result == images

    def test_panel_shuffle_returns_same_elements(self):
        images = list(range(5))
        result = apply_perturbation(images, PerturbationType.PANEL_SHUFFLE)
        assert sorted(result) == sorted(images)
        assert len(result) == len(images)

    def test_all_perturbation_types_run(self):
        images = ["dummy_image"]
        for pert in PerturbationType:
            result = apply_perturbation(images, pert)
            assert isinstance(result, list)

    def test_font_swap_stub_returns_list(self):
        result = apply_perturbation(["img"], PerturbationType.FONT_SWAP)
        assert isinstance(result, list)

    def test_affine_stub_returns_list(self):
        result = apply_perturbation(["img"], PerturbationType.AFFINE)
        assert isinstance(result, list)


class TestSelfPlayMiner:
    def _make_miner(self):
        return SelfPlayMiner(
            student_model=None,
            refresh_interval=2,
            perturbations=[
                PerturbationType.FONT_SWAP,
                PerturbationType.CLUTTER,
            ],
        )

    def test_should_refresh(self):
        miner = self._make_miner()
        assert not miner.should_refresh(0)
        assert miner.should_refresh(2)
        assert not miner.should_refresh(3)
        assert miner.should_refresh(4)

    def test_refresh_adversary_sets_adversary(self):
        import torch.nn as nn

        class DummyModel(nn.Module):
            def forward(self, x):
                return x

        miner = SelfPlayMiner(student_model=DummyModel(), refresh_interval=2)
        assert miner.adversary is None
        miner.refresh_adversary()
        assert miner.adversary is not None

    def test_mine_returns_list(self):
        import torch.nn as nn

        class DummyModel(nn.Module):
            def forward(self, x):
                return x

        miner = SelfPlayMiner(
            student_model=DummyModel(),
            refresh_interval=2,
            perturbations=[PerturbationType.FONT_SWAP],
        )
        examples = [(("imgs", "q"), "answer")]
        result = miner.mine(examples)
        assert isinstance(result, list)

    def test_update_calibration_pools(self):
        miner = self._make_miner()
        from pop.self_play import PerturbedExample
        ex = PerturbedExample(
            original_input=None,
            perturbed_input=None,
            perturbation=PerturbationType.FONT_SWAP,
            correct_output="answer",
            node_level_pairs=[("x_v", "z_v")],
        )
        pools: dict = {}
        miner.update_calibration_pools([ex], pools)
        assert "ocr-string" in pools
        assert len(pools["ocr-string"]) == 1
