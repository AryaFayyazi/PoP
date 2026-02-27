"""
Two-phase inference for PoP (§3.8).

Phase 1 – Graph generation:
    The MLLM planner autoregressively generates a DSL program π.
    The program is parsed into a DAG G = (V, E).

Phase 2 – Graph execution:
    Nodes are executed in topological order σ.  For each node v:
        1. Construct node input x_v from images and upstream conformal sets.
        2. Produce candidate outputs and conformal set Γ^(t)_δ(x_v).
        3. Query the controller πϕ; act on ACCEPT / RETRY / EXPAND / ABORT.
    After visiting all nodes, the answer conformal set Γ^(answer)_δ(x) is
    returned from the answer node v⋆.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from pop.controller import AdaptiveController, BudgetTracker, encode_certificate_state
from pop.dsl import parse_program
from pop.graph import Action, NodeSpec, NodeType, ReasoningGraph
from pop.nodes import BaseNode, NodeRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default DSL program fallback
# ---------------------------------------------------------------------------

_DEFAULT_PROGRAM_TEMPLATE = """\
CALL_TOOL(ocr, img0, "Extract all text from the document") -> v0
FUSE(v0, "Answer the question: {query}") -> v1
RETURN(v1)
"""


def _make_default_program(query: str) -> str:
    return _DEFAULT_PROGRAM_TEMPLATE.format(query=query)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class PoP:
    """
    Inference engine for Proof-of-Perception.

    Parameters
    ----------
    registry:
        NodeRegistry mapping tool keys to BaseNode instances.
    controller:
        Trained AdaptiveController πϕ.
    thresholds:
        Dict mapping NodeType.value → calibrated threshold τ^(t)_δ.
    budget:
        Per-sample compute budget B (default 16, §4.3).
    context_dim:
        Dimension of the context embedding used by the controller.
    greedy:
        If True, the controller uses argmax action selection (for evaluation).
    planner:
        Optional callable(images, query) → program_str.
        If None, the default single-step OCR → fuse template is used.
    """

    def __init__(
        self,
        registry: NodeRegistry,
        controller: AdaptiveController,
        thresholds: Dict[str, float],
        budget: float = 16.0,
        context_dim: int = 256,
        greedy: bool = True,
        planner: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.controller = controller
        self.thresholds = thresholds
        self.budget_limit = budget
        self.context_dim = context_dim
        self.greedy = greedy
        self.planner = planner

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        images: List[Any],
        query: str,
    ) -> Tuple[List[Any], ReasoningGraph]:
        """
        Run the two-phase PoP inference pipeline.

        Parameters
        ----------
        images:
            List of input images (PIL / numpy / tensors).
        query:
            Natural-language task query.

        Returns
        -------
        answer_set:
            Conformal prediction set Γ^(answer)_δ(x) at the answer node.
        graph:
            The executed ReasoningGraph with populated node outputs, actions,
            and costs.
        """
        # ---- Phase 1: graph generation --------------------------------
        graph = self._generate_graph(images, query)
        # ---- Phase 2: graph execution ---------------------------------
        answer_set = self._execute_graph(graph, images, query)
        return answer_set, graph

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _generate_graph(self, images: List[Any], query: str) -> ReasoningGraph:
        """Generate and parse the DSL program into a ReasoningGraph."""
        if self.planner is not None:
            try:
                program = self.planner(images, query)
                return parse_program(program)
            except Exception as exc:
                logger.warning(
                    "Planner failed (%s); falling back to default program.", exc
                )
        # Fallback: single-pass OCR → fuse template
        program = _make_default_program(query)
        return parse_program(program)

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _execute_graph(
        self,
        graph: ReasoningGraph,
        images: List[Any],
        query: str,
    ) -> List[Any]:
        """
        Execute nodes in topological order, applying the controller at each node.
        Returns the conformal set at the answer node.
        """
        budget = BudgetTracker(total_budget=self.budget_limit)
        # Maps node_id → conformal set (list of candidates)
        node_outputs: Dict[str, List[Any]] = {}

        order = graph.topological_order()

        for nid in order:
            if budget.exhausted():
                logger.info("Budget exhausted before node '%s'; aborting.", nid)
                break
            spec = graph.nodes[nid]
            conformal_out, action = self._execute_node(
                spec, graph, images, query, budget, node_outputs
            )
            spec.conformal_set = conformal_out
            spec.action_taken = action
            node_outputs[nid] = conformal_out

            if action == Action.ABORT:
                logger.info("Controller chose ABORT at node '%s'.", nid)
                break

        # Retrieve answer conformal set
        answer_nid = graph.answer_node_id
        return node_outputs.get(answer_nid, [])

    def _execute_node(
        self,
        spec: NodeSpec,
        graph: ReasoningGraph,
        images: List[Any],
        query: str,
        budget: BudgetTracker,
        node_outputs: Dict[str, List[Any]],
    ) -> Tuple[List[Any], Action]:
        """
        Execute a single node and ask the controller what to do next.

        Returns (conformal_set, action_taken).
        """
        threshold = self.thresholds.get(spec.node_type.value, 1.0)
        context_dict = dict(node_outputs)
        context_dict["__query__"] = query

        # --- first attempt ---
        node_impl = self._get_node_impl(spec)
        conformal_out = node_impl.execute(spec, images, threshold, context_dict)

        # Accumulate cost
        cost = (
            BudgetTracker.FUSE_COST if spec.is_fusion else BudgetTracker.TOOL_COST
        )
        budget.spend(cost)
        spec.cost = cost

        # --- controller decision ---
        cert_state = encode_certificate_state(spec, threshold, len(conformal_out))
        budget_tensor = budget.as_tensor()
        ctx_tensor = torch.zeros(self.context_dim)  # placeholder; replace with MLLM embed
        action = self.controller.select_action(
            cert_state, budget_tensor, ctx_tensor, greedy=self.greedy
        )

        if action == Action.RETRY and not budget.exhausted():
            # Re-run with higher-quality config (stub: same call, costs 2 units)
            conformal_out = node_impl.execute(spec, images, threshold, context_dict)
            budget.spend(BudgetTracker.RETRY_COST - BudgetTracker.TOOL_COST)
            spec.cost += BudgetTracker.RETRY_COST - BudgetTracker.TOOL_COST
            action = Action.ACCEPT  # treat retry as accepted

        elif action == Action.EXPAND and not budget.exhausted():
            # Add sub-region child nodes to the graph (stub: log only)
            logger.info("EXPAND requested for node '%s' – sub-region expansion.", spec.node_id)
            action = Action.ACCEPT

        return conformal_out, action

    def _get_node_impl(self, spec: NodeSpec) -> BaseNode:
        """Look up the node implementation from the registry."""
        if spec.is_fusion:
            impl = self.registry.get("fuse")
        else:
            impl = self.registry.get(spec.tool_key or "")
        if impl is None:
            raise KeyError(
                f"No registered node implementation for tool key '{spec.tool_key}'."
            )
        return impl
