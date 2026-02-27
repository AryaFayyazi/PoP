"""
Self-play counterexample mining for PoP (§3.6).

Every ``refresh_interval`` training epochs, a frozen copy of the student model
(adversary) generates perturbed inputs and intermediate states by applying
controlled visual/layout/tool shifts.  The student is trained on selected
counterexamples to recover the correct answer and maintain coverage.

Supported perturbations (matching §3.6 and Table 3):
    - FontSwap   : swap text-rendering font on cropped regions
    - Clutter    : add random bounding-box distractors
    - Affine     : mild affine rotation/shear
    - PanelShuffle : re-order panels in multi-image inputs

Selected hard nodes are appended to the per-type calibration pools C^(t)
so that thresholds τ^(t)_δ reflect realistic failure modes (Eq. 13–15).
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Perturbation types
# ---------------------------------------------------------------------------


class PerturbationType(str, Enum):
    NONE = "None"
    FONT_SWAP = "FontSwap"
    CLUTTER = "Clutter10%"
    AFFINE = "Affine(3°)"
    PANEL_SHUFFLE = "PanelShuffle"


@dataclass
class PerturbedExample:
    """Container for a perturbed input and its known correct output."""
    original_input: Any          # (images, query) tuple
    perturbed_input: Any         # perturbed version
    perturbation: PerturbationType
    correct_output: Any          # ground-truth answer y
    node_level_pairs: List[Tuple[Any, Any]] = field(default_factory=list)
    # (x_v', z_v') pairs to add to calibration pools


# ---------------------------------------------------------------------------
# Perturbation functions (image-level stubs)
# ---------------------------------------------------------------------------


def _font_swap(images: List[Any]) -> List[Any]:
    """Swap rendered text fonts in image crops (stub)."""
    return images  # Replace with PIL font manipulation


def _add_clutter(images: List[Any], fraction: float = 0.10) -> List[Any]:
    """Add random bounding-box distractor rectangles (stub)."""
    return images  # Replace with cv2 / PIL rectangle drawing


def _affine_warp(images: List[Any], degrees: float = 3.0) -> List[Any]:
    """Apply a mild affine rotation to each image (stub)."""
    return images  # Replace with torchvision.transforms.RandomAffine


def _panel_shuffle(images: List[Any]) -> List[Any]:
    """Randomly re-order panels in a multi-image input (stub)."""
    shuffled = images.copy()
    random.shuffle(shuffled)
    return shuffled


_PERTURBATION_FN: Dict[PerturbationType, Callable] = {
    PerturbationType.NONE: lambda imgs: imgs,
    PerturbationType.FONT_SWAP: _font_swap,
    PerturbationType.CLUTTER: _add_clutter,
    PerturbationType.AFFINE: _affine_warp,
    PerturbationType.PANEL_SHUFFLE: _panel_shuffle,
}


def apply_perturbation(
    images: List[Any],
    perturbation: PerturbationType,
) -> List[Any]:
    fn = _PERTURBATION_FN.get(perturbation, lambda imgs: imgs)
    return fn(images)


# ---------------------------------------------------------------------------
# Self-play mining loop
# ---------------------------------------------------------------------------


class SelfPlayMiner:
    """
    Wraps the student model with a periodically frozen adversary copy (§3.6).

    Usage
    -----
    >>> miner = SelfPlayMiner(student_model, refresh_interval=2)
    >>> for epoch in range(num_epochs):
    ...     if miner.should_refresh(epoch):
    ...         miner.refresh_adversary()
    ...     counterexamples = miner.mine(batch)
    ...     student_model.train_on(counterexamples)

    Parameters
    ----------
    student_model:
        The trainable PoP model (must expose ``run_graph`` and ``calibration_pools``).
    refresh_interval:
        Number of epochs between adversary refreshes (default 2, §4.3).
    perturbations:
        Which perturbation types to apply (defaults to all from the paper).
    nonconformity_threshold:
        A node is selected as a hard counterexample if its nonconformity score
        exceeds this fraction of its calibrated threshold τ^(t)_δ.
    """

    def __init__(
        self,
        student_model: Any,
        refresh_interval: int = 2,
        perturbations: Optional[Sequence[PerturbationType]] = None,
        nonconformity_threshold: float = 0.8,
    ) -> None:
        self.student = student_model
        self.refresh_interval = refresh_interval
        self.perturbations = list(perturbations) if perturbations else [
            PerturbationType.FONT_SWAP,
            PerturbationType.CLUTTER,
            PerturbationType.AFFINE,
            PerturbationType.PANEL_SHUFFLE,
        ]
        self.nonconformity_threshold = nonconformity_threshold
        self.adversary: Optional[Any] = None

    def should_refresh(self, epoch: int) -> bool:
        return epoch > 0 and epoch % self.refresh_interval == 0

    def refresh_adversary(self) -> None:
        """Create a deep-frozen copy of the current student model."""
        self.adversary = copy.deepcopy(self.student)
        for param in self.adversary.parameters():
            param.requires_grad_(False)

    def mine(
        self,
        examples: List[Tuple[Any, Any]],
    ) -> List[PerturbedExample]:
        """
        Generate and filter counterexamples for a batch.

        Parameters
        ----------
        examples:
            List of (input_x, label_y) tuples.

        Returns
        -------
        List of PerturbedExample where the adversary made an error or produced
        high nonconformity on at least one node.
        """
        if self.adversary is None:
            self.refresh_adversary()

        results: List[PerturbedExample] = []
        for x, y in examples:
            images, query = x if isinstance(x, tuple) else (x, "")
            for pert in self.perturbations:
                perturbed_images = apply_perturbation(list(images), pert)
                perturbed_x = (perturbed_images, query)
                # Run the adversary on perturbed input
                adv_output, node_pairs = self._run_adversary(perturbed_x)
                # Filter: keep if adversary is wrong or any node is hard
                if self._is_counterexample(adv_output, y, node_pairs):
                    results.append(
                        PerturbedExample(
                            original_input=x,
                            perturbed_input=perturbed_x,
                            perturbation=pert,
                            correct_output=y,
                            node_level_pairs=node_pairs,
                        )
                    )
        return results

    def _run_adversary(
        self,
        perturbed_x: Any,
    ) -> Tuple[Any, List[Tuple[Any, Any]]]:
        """
        Run the frozen adversary on a perturbed input.

        Returns
        -------
        output:
            Adversary's final answer (may be wrong).
        node_pairs:
            List of (x_v', z_v') pairs from intermediate nodes.
        """
        # Stub: replace with adversary.run_graph(perturbed_x)
        return None, []

    def _is_counterexample(
        self,
        adv_output: Any,
        correct_output: Any,
        node_pairs: List[Tuple[Any, Any]],
    ) -> bool:
        """
        A hard example is one where the adversary is wrong or any node has
        high nonconformity (> nonconformity_threshold × τ^(t)_δ).
        """
        if adv_output != correct_output:
            return True
        # In a full implementation, check node-level nonconformity scores here.
        return False

    def update_calibration_pools(
        self,
        counterexamples: List[PerturbedExample],
        calibration_pools: Dict[str, List[Tuple[Any, Any]]],
    ) -> None:
        """
        Append selected node-level pairs to per-type calibration pools C^(t).

        Parameters
        ----------
        counterexamples:
            Hard examples identified by ``mine()``.
        calibration_pools:
            Mutable dict mapping NodeType value → list of (x_v', z_v') pairs.
        """
        for ex in counterexamples:
            for x_v, z_v in ex.node_level_pairs:
                # In a full implementation, determine the node type here.
                node_type_key = "ocr-string"  # placeholder
                if node_type_key not in calibration_pools:
                    calibration_pools[node_type_key] = []
                calibration_pools[node_type_key].append((x_v, z_v))
