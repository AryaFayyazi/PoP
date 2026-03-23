"""
Reasoning graph representation for Proof-of-Perception (PoP).

A reasoning graph G = (V, E) is a directed acyclic graph where each node v
corresponds to either a perception tool call or a logic-fusion operation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class NodeType(str, enum.Enum):
    """Supported node types, corresponding to nonconformity score families."""
    OCR = "ocr-string"
    DETECTION = "det-box"
    CHART = "chart-num"
    LOGIC = "logic-text"


class Action(str, enum.Enum):
    """Discrete action space for the adaptive controller (§3.5)."""
    ACCEPT = "ACCEPT"
    RETRY = "RETRY"
    EXPAND = "EXPAND"
    ABORT = "ABORT"


@dataclass
class NodeSpec:
    """
    Specification for a single reasoning node.

    Tool nodes call an external perception tool; fusion nodes run a forward
    pass through the MLLM to combine intermediate results.
    """
    node_id: str
    node_type: NodeType
    # --- tool node fields ---
    tool_key: Optional[str] = None          # e.g. "ocr", "detector", "chart"
    region: Optional[Any] = None            # image index + bounding box
    prompt: Optional[str] = None
    # --- fusion node fields ---
    is_fusion: bool = False
    # --- shared ---
    parents: List[str] = field(default_factory=list)
    # runtime results (populated during execution)
    output: Optional[Any] = None           # point prediction ˆz
    conformal_set: Optional[List[Any]] = None  # Γ^(t)_δ(x_v)
    nonconformity_score: Optional[float] = None
    action_taken: Optional[Action] = None
    cost: float = 0.0


@dataclass
class ReasoningGraph:
    """
    A directed acyclic graph (DAG) over NodeSpec objects.

    Nodes are identified by string IDs. Edges are encoded as parent lists
    inside each NodeSpec.  The graph exposes helpers for topological ordering,
    acyclicity checks, and dynamic node insertion (for EXPAND actions).
    """

    nodes: Dict[str, NodeSpec] = field(default_factory=dict)
    answer_node_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def add_node(self, spec: NodeSpec) -> None:
        """Register a node; raises if a duplicate ID is inserted."""
        if spec.node_id in self.nodes:
            raise ValueError(f"Node '{spec.node_id}' already exists in graph.")
        # validate parents exist
        for p in spec.parents:
            if p not in self.nodes:
                raise ValueError(
                    f"Parent node '{p}' of '{spec.node_id}' not yet in graph."
                )
        self.nodes[spec.node_id] = spec
        if not self._is_acyclic():
            self.nodes.pop(spec.node_id)
            raise ValueError(
                f"Adding node '{spec.node_id}' would introduce a cycle."
            )

    def set_answer_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise ValueError(f"Unknown node id '{node_id}'.")
        self.answer_node_id = node_id

    # ------------------------------------------------------------------
    # Graph algorithms
    # ------------------------------------------------------------------

    def topological_order(self) -> List[str]:
        """Return node IDs in a valid topological order (Kahn's algorithm)."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for spec in self.nodes.values():
            for p in spec.parents:
                in_degree[spec.node_id] = in_degree[spec.node_id] + 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: List[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for other_id, other_spec in self.nodes.items():
                if nid in other_spec.parents:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)
        if len(order) != len(self.nodes):
            raise RuntimeError("Cycle detected in reasoning graph.")
        return order

    def _is_acyclic(self) -> bool:
        try:
            self.topological_order()
            return True
        except RuntimeError:
            return False

    def children(self, node_id: str) -> List[str]:
        """Return IDs of all direct children of a given node."""
        return [
            nid for nid, spec in self.nodes.items()
            if node_id in spec.parents
        ]

    def total_cost(self) -> float:
        return sum(spec.cost for spec in self.nodes.values())

    def __repr__(self) -> str:  # pragma: no cover
        lines = [f"ReasoningGraph(answer={self.answer_node_id})"]
        for order_idx, nid in enumerate(self.topological_order()):
            spec = self.nodes[nid]
            lines.append(
                f"  [{order_idx}] {nid} type={spec.node_type.value}"
                f" parents={spec.parents} action={spec.action_taken}"
            )
        return "\n".join(lines)
