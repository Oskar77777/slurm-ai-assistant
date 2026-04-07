"""
Resource planner for GPU allocation on the eX3 cluster.
Deterministically computes feasible partition/node combinations for a given GPU request,
so the LLM never has to do this arithmetic itself.
"""

import math
import re
from typing import Any


def _build_partition_gpu_map(nodes_data: dict[str, Any]) -> dict:
    """Build a map of (partition, gpu_model, gpus_per_node) -> availability info."""
    partition_map = {}

    for name, node in nodes_data.items():
        partitions = node.get("partitions", [])
        cards = node.get("cards", [])
        if not partitions or not cards:
            continue

        total_gpus = len(cards)
        alloc = node.get("alloc_tres", {})
        available_gpus = total_gpus - alloc.get("gpu", 0)
        is_idle = alloc.get("node", 0) == 0
        gpu_model = cards[0].get("model", "Unknown").replace("NVIDIA ", "")

        for partition in partitions:
            key = (partition, gpu_model, total_gpus)
            if key not in partition_map:
                partition_map[key] = {"idle_nodes": 0, "total_nodes": 0, "available_gpus": 0}
            entry = partition_map[key]
            entry["total_nodes"] += 1
            entry["available_gpus"] += available_gpus
            if is_idle:
                entry["idle_nodes"] += 1

    return partition_map


def recommend_gpu_allocation(nodes_data: dict[str, Any], num_gpus: int) -> str:
    """
    Given nodes data and a requested GPU count, return a pre-computed recommendation
    showing exactly which partitions can satisfy the request and how many nodes to use.
    """
    partition_map = _build_partition_gpu_map(nodes_data)

    feasible = []
    infeasible = []

    for (partition, gpu_model, gpus_per_node), info in sorted(partition_map.items()):
        nodes_needed = math.ceil(num_gpus / gpus_per_node)
        # Request only the GPUs actually needed per node, not the full node allocation
        gpus_per_node_requested = math.ceil(num_gpus / nodes_needed)
        entry = {
            "partition": partition,
            "gpu_model": gpu_model,
            "gpus_per_node": gpus_per_node,
            "gpus_per_node_requested": gpus_per_node_requested,
            "nodes_needed": nodes_needed,
            "idle_nodes": info["idle_nodes"],
            "total_nodes": info["total_nodes"],
            "available_gpus": info["available_gpus"],
        }
        if info["idle_nodes"] >= nodes_needed:
            feasible.append(entry)
        else:
            infeasible.append(entry)

    lines = [f"[RESOURCE RECOMMENDATION FOR {num_gpus} GPUs]"]
    lines.append("GPU PARTITION SUMMARY:")
    lines.append("  Format: partition | GPU model | GPUs/node | nodes (idle/total) | GPUs available/total")
    for (partition, gpu_model, gpus_per_node), info in sorted(partition_map.items()):
        lines.append(
            f"  {partition}: {gpu_model}, {gpus_per_node} GPU/node, "
            f"{info['idle_nodes']}/{info['total_nodes']} nodes idle, "
            f"{info['available_gpus']} GPUs available"
        )
    lines.append("")

    if feasible:
        lines.append("FEASIBLE OPTIONS (use one of these for --partition and --nodes):")
        for e in feasible:
            lines.append(
                f"  {e['partition']}: {e['gpu_model']}, "
                f"--nodes={e['nodes_needed']} --gpus-per-node={e['gpus_per_node_requested']} "
                f"({e['idle_nodes']}/{e['total_nodes']} nodes currently idle)"
            )
    else:
        lines.append(f"NO FEASIBLE OPTIONS: No partition has enough idle nodes for {num_gpus} GPUs right now.")

    if infeasible:
        lines.append("NOT FEASIBLE (not enough idle nodes):")
        for e in infeasible:
            max_gpus = e["idle_nodes"] * e["gpus_per_node"]
            lines.append(
                f"  {e['partition']}: {e['idle_nodes']}/{e['total_nodes']} nodes idle "
                f"= max {max_gpus} GPUs available (need {e['nodes_needed']} nodes for {num_gpus} GPUs)"
            )

    return "\n".join(lines)


def detect_node_query_intent(messages: list[dict]) -> str:
    """
    Detect whether the user is asking about GPU nodes, CPU nodes, or all nodes.
    Returns "gpu", "cpu", or "all".
    """
    gpu_keywords = ["gpu", "graphics", "cuda", "a100", "v100", "h100", "h200", "a40",
                    "mi210", "mi100", "mi50", "gh200", "accelerat"]
    cpu_keywords = ["cpu", "cpu-only", "cpu only", "compute node", "no gpu", "without gpu"]

    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "").lower()
        if any(kw in content for kw in cpu_keywords):
            return "cpu"
        if any(kw in content for kw in gpu_keywords):
            return "gpu"

    return "all"


def extract_gpu_count(messages: list[dict]) -> int | None:
    """Extract a GPU count from the most recent user messages, if present."""
    patterns = [
        r"(\d+)\s*gpu",
        r"(\d+)\s*graphics card",
        r"requires?\s+(\d+)",
        r"need\s+(\d+)",
    ]
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "").lower()
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return int(match.group(1))
    return None
