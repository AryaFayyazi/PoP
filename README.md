# Proof-of-Perception (PoP)

**Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees**

> CVPR 2026 

---

## Overview

PoP casts multimodal reasoning as the execution of a **directed acyclic graph (DAG)** whose
nodes are perception or logic operations, each equipped with a **conformal prediction
certificate**.  A lightweight **adaptive controller** observes per-node uncertainty sets and a
compute budget to decide when to accept, retry, expand with additional tool calls, or abort
early.

This design:
- **Grounds answers** in verifiable perceptual evidence (OCR, detection, chart parsing)
- **Reduces error compounding and hallucinations** by keeping multiple calibrated candidates
  until evidence resolves ambiguity
- **Enables principled accuracy–compute trade-offs** by tying computation to certified
  uncertainty

---

## Architecture

```
              ┌──────────────────────────────────────────┐
  Images +    │           Planner (MLLM)                 │
  Text Prompt │  generates DSL program π → DAG G=(V,E)   │
              └──────────────────────┬───────────────────┘
                                     │
              ┌──────────────────────▼───────────────────┐
              │         Graph Execution (topological)    │
              │                                          │
              │  ┌─────────┐   ┌──────────┐   ┌───────┐ │
              │  │ OCR     │   │ Detect.  │   │ Chart │ │
              │  │ Node    │   │ Node     │   │ Node  │ │
              │  │ Γ^ocr_δ │   │ Γ^det_δ  │   │ Γ^ch_δ│ │
              │  └────┬────┘   └────┬─────┘   └───┬───┘ │
              │       └─────────────┴─────────────┘     │
              │                     │                    │
              │            ┌────────▼────────┐           │
              │            │  Logic Fusion   │           │
              │            │  Node  Γ^lg_δ   │           │
              │            └────────┬────────┘           │
              │                     │                    │
              │       ┌─────────────▼──────────────┐     │
              │       │   Adaptive Controller πϕ   │     │
              │       │  {ACCEPT, RETRY, EXPAND,   │     │
              │       │           ABORT}            │     │
              │       └─────────────┬──────────────┘     │
              └─────────────────────┼────────────────────┘
                                    │
                        ┌───────────▼──────────┐
                        │  Certified Answer    │
                        │  Γ^(answer)_δ(x)     │
                        └──────────────────────┘
```

### Key Components

| Module | Description |
|---|---|
| `pop/graph.py` | DAG representation (`ReasoningGraph`, `NodeSpec`, `NodeType`, `Action`) |
| `pop/dsl.py` | DSL parser / serialiser (`CALL_TOOL`, `FUSE`, `RETURN`) |
| `pop/conformal.py` | Split-CP calibration, conformal sets, certificate head |
| `pop/nodes/` | OCR, Detection, Chart, Logic-Fusion node implementations |
| `pop/controller.py` | Adaptive controller πϕ + REINFORCE loss |
| `pop/self_play.py` | Self-play counterexample mining |
| `pop/training.py` | Combined loss L = L_task + L_plan + L_cert + L_ctrl |
| `pop/inference.py` | Two-phase inference engine |
| `pop/model.py` | Top-level `ProofOfPerception` class |

---

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

---

## Quick Start

```python
from pop import ProofOfPerception

model = ProofOfPerception(
    feature_dim=256,
    context_dim=256,
    budget=16.0,
    delta=0.1,         # 90% coverage target
)

# Calibrate node thresholds from held-out calibration set
from pop.graph import NodeType
for score in calibration_scores["ocr"]:
    model.add_calibration_score(NodeType.OCR, score)
model.calibrate_nodes()

# Run inference
answer_set, graph = model.infer(images=my_images, query="What is the total revenue?")
# answer_set: Γ^(answer)_0.1(x) – calibrated set of answer candidates
print(answer_set)
print(graph)  # full execution trace with per-node actions and costs
```

### Custom DSL Programs (manual graph control)

```python
program = """
CALL_TOOL(ocr, img0, "Extract all text") -> v0
CALL_TOOL(chart, img1[0,0,500,400], "Read chart values") -> v1
FUSE(v0, v1, "Summarise and answer the question") -> v2
RETURN(v2)
"""

answer_set, graph = model.infer(
    images=[doc_image, chart_image],
    query="What is the chart's peak value?",
    planner=lambda imgs, q: program,
)
```

### Integrating Real Tools

The stub implementations in `pop/nodes/__init__.py` expose protected hooks that
can be overridden in subclasses without touching the conformal prediction logic:

```python
import paddleocr
from pop.nodes import OCRNode

class PaddleOCRNode(OCRNode):
    def __init__(self):
        self.ocr_engine = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")
        super().__init__()

    def _call_ocr(self, spec, images):
        img = images[spec.region["image_index"]]
        result = self.ocr_engine.ocr(img, cls=True)
        texts = [line[1][0] for block in result for line in block]
        confs = [line[1][1] for block in result for line in block]
        return texts, confs  # (candidates, probs)

model.registry.register("ocr", PaddleOCRNode())
```

Similarly, `DetectionNode` exposes `_call_detector(self, spec, images) -> List[List[float]]`
(boxes ordered by descending confidence) and `ChartNode` exposes
`_call_chart_parser(self, spec, images) -> Tuple[float, List[float]]`
(predicted mean and candidate values).

---

## Conformal Prediction Details (§3.4)

For each node type t, PoP uses **split conformal prediction**:

1. Collect calibration nonconformity scores  αⱼ = s⁽ᵗ⁾(xⱼ, zⱼ)
2. Select threshold  τ⁽ᵗ⁾_δ = α₍ₖ₎,  k = ⌈(n+1)(1−δ)⌉
3. At test time:  Γ⁽ᵗ⁾_δ(xᵥ) = { z : s⁽ᵗ⁾(xᵥ, z) ≤ τ⁽ᵗ⁾_δ }

This guarantees  P(z_true ∈ Γ⁽ᵗ⁾_δ(xᵥ)) ≥ 1 − δ  under exchangeability.

Default nonconformity score families:

| Node type | Score |
|---|---|
| `ocr-string` | 1 − P_θ(z \| xᵥ) |
| `det-box` | 1 − IoU(z, ẑ_MAP) |
| `chart-num` | \|z − μ_θ(xᵥ)\| |
| `logic-text` | certificate head output |

---

## Training (§3.7)

```
L = L_task + γ_plan · L_plan + γ_cert · L_cert + γ_ctrl · L_ctrl
```

| Loss | Purpose |
|---|---|
| `L_task` | Cross-entropy / Smooth-L1 on final answer |
| `L_plan` | NLL for planner DSL program |
| `L_cert` | Margin loss: push s⁽ᵗ⁾(xᵥ, zᵥ) ≤ τ⁽ᵗ⁾_δ for true outputs |
| `L_ctrl` | REINFORCE: minimise C_err + β · C_comp |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Datasets Evaluated (§4.1)

| Task | Dataset | Metric |
|---|---|---|
| Document QA | DocVQA (Task 1 & 3) | EM, Hallucination Rate |
| OCR-heavy QA | TextVQA | F1, Hallucination Rate |
| Dense graphics | InfographicVQA | EM |
| Chart reasoning | ChartQA (Human/Machine) | EM, Abs. Error |
| Multi-image QA | MultiDoc2Dial | EM |
| Captioning | TextCaps | CIDEr |

---

## Citation

```
Coming soon!
```
