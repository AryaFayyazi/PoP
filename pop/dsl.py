"""
Domain-specific language (DSL) for encoding PoP reasoning graphs.

The planner MLLM autoregressively emits a text program in this DSL which
is then parsed into a ReasoningGraph by a deterministic interpreter.

Grammar (one instruction per line)
-----------------------------------
    CALL_TOOL(<tool_key>, <region>, <prompt>)  -> <node_id>
    FUSE(<parent1>[,<parent2>,...], <prompt>)   -> <node_id>
    RETURN(<node_id>)

Region format: "img<index>[x1,y1,x2,y2]" or "img<index>" (full image).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from pop.graph import NodeSpec, NodeType, ReasoningGraph

# Mapping from tool key to node type
_TOOL_TYPE_MAP: Dict[str, NodeType] = {
    "ocr": NodeType.OCR,
    "detector": NodeType.DETECTION,
    "det": NodeType.DETECTION,
    "chart": NodeType.CHART,
    "layout": NodeType.LOGIC,
    "vqa": NodeType.LOGIC,
}

_RE_CALL_TOOL = re.compile(
    r"CALL_TOOL\(\s*(?P<tool>\w+)\s*,\s*(?P<region>\w+(?:\[[^\]]*\])?)\s*,\s*(?P<prompt>.+?)\s*\)\s*->\s*(?P<nid>\w+)",
    re.IGNORECASE,
)
_RE_FUSE = re.compile(
    r"FUSE\(\s*(?P<parents>\w+(?:\s*,\s*\w+)*)\s*,\s*(?P<prompt>.+?)\s*\)\s*->\s*(?P<nid>\w+)",
    re.IGNORECASE,
)
_RE_RETURN = re.compile(r"RETURN\(\s*(?P<nid>\w+)\s*\)", re.IGNORECASE)

_RE_REGION = re.compile(
    r"img(?P<idx>\d+)(?:\[(?P<x1>[\d.]+),(?P<y1>[\d.]+),(?P<x2>[\d.]+),(?P<y2>[\d.]+)\])?",
    re.IGNORECASE,
)


def _parse_region(raw: str) -> dict:
    """Return a dict with image index and optional bounding box."""
    m = _RE_REGION.match(raw.strip())
    if not m:
        return {"image_index": 0, "bbox": None}
    result: dict = {"image_index": int(m.group("idx"))}
    if m.group("x1") is not None:
        result["bbox"] = (
            float(m.group("x1")),
            float(m.group("y1")),
            float(m.group("x2")),
            float(m.group("y2")),
        )
    else:
        result["bbox"] = None
    return result


def _infer_node_type(tool_key: str, is_fusion: bool) -> NodeType:
    if is_fusion:
        return NodeType.LOGIC
    key = tool_key.lower()
    for k, nt in _TOOL_TYPE_MAP.items():
        if k in key:
            return nt
    return NodeType.LOGIC


def parse_program(program: str) -> ReasoningGraph:
    """
    Parse a DSL program string into a ReasoningGraph.

    Parameters
    ----------
    program:
        Multi-line string where each line is a DSL instruction.

    Returns
    -------
    ReasoningGraph
        A validated, acyclic DAG ready for execution.

    Raises
    ------
    ValueError
        On any syntax or semantic error (unknown node reference, cycle, etc.).
    """
    graph = ReasoningGraph()
    lines = [ln.strip() for ln in program.splitlines() if ln.strip()]

    for lineno, line in enumerate(lines, start=1):
        # --- CALL_TOOL ---
        m = _RE_CALL_TOOL.match(line)
        if m:
            tool_key = m.group("tool")
            region = _parse_region(m.group("region"))
            prompt = m.group("prompt").strip('"\'')
            nid = m.group("nid")
            node_type = _infer_node_type(tool_key, is_fusion=False)
            spec = NodeSpec(
                node_id=nid,
                node_type=node_type,
                tool_key=tool_key,
                region=region,
                prompt=prompt,
                is_fusion=False,
                parents=[],
            )
            graph.add_node(spec)
            continue

        # --- FUSE ---
        m = _RE_FUSE.match(line)
        if m:
            raw_parents = [p.strip() for p in m.group("parents").split(",")]
            prompt = m.group("prompt").strip('"\'')
            nid = m.group("nid")
            for p in raw_parents:
                if p not in graph.nodes:
                    raise ValueError(
                        f"Line {lineno}: parent '{p}' not defined before FUSE."
                    )
            spec = NodeSpec(
                node_id=nid,
                node_type=NodeType.LOGIC,
                is_fusion=True,
                prompt=prompt,
                parents=raw_parents,
            )
            graph.add_node(spec)
            continue

        # --- RETURN ---
        m = _RE_RETURN.match(line)
        if m:
            nid = m.group("nid")
            graph.set_answer_node(nid)
            continue

        raise ValueError(f"Line {lineno}: unrecognized DSL instruction: {line!r}")

    if graph.answer_node_id is None:
        raise ValueError("Program missing RETURN instruction.")
    return graph


def graph_to_program(graph: ReasoningGraph) -> str:
    """
    Serialise a ReasoningGraph back to a DSL program string.

    Useful for logging or prompt construction.
    """
    lines: List[str] = []
    for nid in graph.topological_order():
        spec = graph.nodes[nid]
        if spec.is_fusion:
            parents_str = ", ".join(spec.parents)
            lines.append(f'FUSE({parents_str}, "{spec.prompt}") -> {nid}')
        else:
            region = spec.region or {}
            idx = region.get("image_index", 0)
            bbox = region.get("bbox")
            if bbox:
                region_str = f"img{idx}[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
            else:
                region_str = f"img{idx}"
            lines.append(
                f'CALL_TOOL({spec.tool_key}, {region_str}, "{spec.prompt}") -> {nid}'
            )
    if graph.answer_node_id:
        lines.append(f"RETURN({graph.answer_node_id})")
    return "\n".join(lines)
