"""
Proof-of-Perception (PoP) — certified tool-using multimodal reasoning
with compositional conformal guarantees.

Public API
----------
>>> from pop import ProofOfPerception
>>> model = ProofOfPerception()
>>> answer_set, graph = model.infer(images=[...], query="What is the total?")
"""

from pop.model import ProofOfPerception
from pop.graph import ReasoningGraph, NodeSpec, NodeType, Action
from pop.conformal import calibrate, conformal_set, CoverageTracker
from pop.controller import AdaptiveController, BudgetTracker
from pop.dsl import parse_program, graph_to_program
from pop.nodes import NodeRegistry, BaseNode, OCRNode, DetectionNode, ChartNode, LogicFusionNode
from pop.inference import PoP
from pop.training import PoPLoss, TaskLoss, PlanningLoss, CertificateLoss
from pop.self_play import SelfPlayMiner, PerturbationType

__all__ = [
    # Main model
    "ProofOfPerception",
    # Graph primitives
    "ReasoningGraph",
    "NodeSpec",
    "NodeType",
    "Action",
    # Conformal prediction
    "calibrate",
    "conformal_set",
    "CoverageTracker",
    # Controller
    "AdaptiveController",
    "BudgetTracker",
    # DSL
    "parse_program",
    "graph_to_program",
    # Nodes
    "NodeRegistry",
    "BaseNode",
    "OCRNode",
    "DetectionNode",
    "ChartNode",
    "LogicFusionNode",
    # Inference
    "PoP",
    # Training
    "PoPLoss",
    "TaskLoss",
    "PlanningLoss",
    "CertificateLoss",
    # Self-play
    "SelfPlayMiner",
    "PerturbationType",
]
