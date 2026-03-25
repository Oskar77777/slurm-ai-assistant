"""
Data processor for eX3 cluster API responses.
Summarizes raw node data into compact format with calculated availability.
"""

from typing import Any


def _build_nodes_by_status(nodes_data: dict[str, Any]) -> dict:
    """Parse raw node data into structured dicts grouped by availability status."""
    nodes_by_status = {"idle": [], "partial": [], "full": []}

    for name, node in nodes_data.items():
        partitions = node.get("partitions", [])
        if not partitions:
            continue

        sockets = node.get("sockets", 1)
        cores_per_socket = node.get("cores_per_socket", 1)
        threads_per_core = node.get("threads_per_core", 1)
        total_cpus = sockets * cores_per_socket * threads_per_core

        cards = node.get("cards", [])
        total_gpus = len(cards)

        alloc = node.get("alloc_tres", {})
        allocated_cpus = alloc.get("cpu", 0)
        allocated_gpus = alloc.get("gpu", 0)

        available_cpus = total_cpus - allocated_cpus
        available_gpus = total_gpus - allocated_gpus

        if alloc.get("node", 0) == 0:
            status = "idle"
        elif available_cpus == 0 and (total_gpus == 0 or available_gpus == 0):
            status = "full"
        else:
            status = "partial"

        gpu_model = None
        if cards:
            gpu_model = cards[0].get("model", "").replace("NVIDIA ", "")

        memory_gb = int(round(node.get("memory", 0) / 1024 / 1024, 0))

        node_info = {
            "name": name,
            "type": "GPU" if total_gpus > 0 else "CPU",
            "cpus": f"{available_cpus}/{total_cpus}",
            "mem_gb": memory_gb,
            "partitions": partitions,
        }
        if total_gpus > 0:
            node_info["gpus"] = f"{available_gpus}/{total_gpus}"
            node_info["gpu_model"] = gpu_model

        nodes_by_status[status].append(node_info)

    for status in nodes_by_status:
        nodes_by_status[status].sort(key=lambda x: x["name"])

    return nodes_by_status


def _format_node_line(n: dict) -> str:
    if n["type"] == "GPU":
        return f"  {n['name']}: {n['cpus']} CPUs, {n['gpus']} {n['gpu_model']} GPUs, {n['mem_gb']} GB RAM [{', '.join(n['partitions'])}]"
    return f"  {n['name']}: {n['cpus']} CPUs, {n['mem_gb']} GB RAM [{', '.join(n['partitions'])}]"


def _format_output(nodes_by_status: dict, node_filter: str = "all") -> str:
    """
    Format nodes_by_status into human-readable text.
    node_filter: "all" | "gpu" | "cpu"
    """
    def keep(n):
        if node_filter == "gpu":
            return n["type"] == "GPU"
        if node_filter == "cpu":
            return n["type"] == "CPU"
        return True

    idle = [n for n in nodes_by_status["idle"] if keep(n)]
    partial = [n for n in nodes_by_status["partial"] if keep(n)]
    full = [n for n in nodes_by_status["full"] if keep(n)]

    total = len(idle) + len(partial) + len(full)
    lines = []

    if node_filter == "gpu":
        lines.append(f"GPU NODES: {total} total")
    elif node_filter == "cpu":
        lines.append(f"CPU NODES: {total} total")
    else:
        all_idle = nodes_by_status["idle"]
        gpu_idle = [n for n in all_idle if n["type"] == "GPU"]
        cpu_idle = [n for n in all_idle if n["type"] == "CPU"]
        total_all = sum(len(v) for v in nodes_by_status.values())
        lines.append(f"CLUSTER STATUS: {total_all} compute nodes total")
        lines.append(f"  - {len(nodes_by_status['idle'])} idle ({len(gpu_idle)} GPU, {len(cpu_idle)} CPU)")
        lines.append(f"  - {len(nodes_by_status['partial'])} partially used")
        lines.append(f"  - {len(nodes_by_status['full'])} fully occupied")

    lines.append(f"  - {len(idle)} idle, {len(partial)} partially used, {len(full)} fully occupied")
    lines.append("")

    if idle:
        lines.append(f"IDLE ({len(idle)}):")
        for n in idle:
            lines.append(_format_node_line(n))

    if partial:
        lines.append("")
        lines.append(f"PARTIAL ({len(partial)}):")
        for n in partial:
            lines.append(_format_node_line(n))

    if full:
        lines.append("")
        lines.append(f"FULLY OCCUPIED ({len(full)}):")
        for n in full:
            if n["type"] == "GPU":
                lines.append(f"  {n['name']}: {n['gpus']} {n['gpu_model']} GPUs [{', '.join(n['partitions'])}]")
            else:
                lines.append(f"  {n['name']}: [{', '.join(n['partitions'])}]")

    return "\n".join(lines)


def summarize_nodes(nodes_data: dict[str, Any]) -> str:
    """Summarize all compute nodes."""
    return _format_output(_build_nodes_by_status(nodes_data), node_filter="all")


def summarize_gpu_nodes(nodes_data: dict[str, Any]) -> str:
    """Summarize GPU nodes only."""
    return _format_output(_build_nodes_by_status(nodes_data), node_filter="gpu")


def summarize_cpu_nodes(nodes_data: dict[str, Any]) -> str:
    """Summarize CPU-only nodes."""
    return _format_output(_build_nodes_by_status(nodes_data), node_filter="cpu")
