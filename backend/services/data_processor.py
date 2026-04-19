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
    return f"  {n['name']}: {n['cpus']} CPUs, no GPU, {n['mem_gb']} GB RAM [{', '.join(n['partitions'])}]"


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


def summarize_partitions(partitions_data: list[Any]) -> str:
    """Summarize partition data into human-readable text."""
    if not partitions_data:
        return "No partition data available."

    lines = [f"PARTITIONS: {len(partitions_data)} total", ""]
    for p in sorted(partitions_data, key=lambda x: x.get("name", "")):
        name = p.get("name", "unknown")
        nodes_compact = p.get("nodes_compact") or p.get("nodes", [])
        nodes_str = ", ".join(nodes_compact) if nodes_compact else "none"
        node_count = len(p.get("nodes", []))
        total_cpus = p.get("total_cpus", 0)
        total_gpus = p.get("total_gpus", 0)
        gpus_reserved = p.get("gpus_reserved", 0)
        gpus_in_use = len(p.get("gpus_in_use", []))
        running = len(p.get("jobs_running", []))
        pending = len(p.get("jobs_pending", []))

        gpu_str = f"  |  {total_gpus} GPUs ({gpus_reserved} reserved, {gpus_in_use} in use)" if total_gpus > 0 else ""
        lines.append(
            f"  {name}: {node_count} node(s) [{nodes_str}]"
            f"  |  {total_cpus} CPUs{gpu_str}"
            f"  |  {running} running, {pending} pending"
        )

    return "\n".join(lines)


def format_single_node(node_data: dict[str, Any]) -> str:
    """Format detailed info for a single node into human-readable text."""
    if not node_data:
        return "No data returned for this node."

    name = list(node_data.keys())[0]
    node = node_data[name]

    sockets = node.get("sockets", 1)
    cores_per_socket = node.get("cores_per_socket", 1)
    threads_per_core = node.get("threads_per_core", 1)
    total_cpus = sockets * cores_per_socket * threads_per_core

    alloc = node.get("alloc_tres", {})
    allocated_cpus = alloc.get("cpu", 0)
    available_cpus = total_cpus - allocated_cpus

    memory_gb = int(round(node.get("memory", 0) / 1024 / 1024, 0))
    allocated_mem_gb = int(round(alloc.get("memory", 0) / 1024 / 1024, 0))
    available_mem_gb = memory_gb - allocated_mem_gb

    cards = node.get("cards", [])
    total_gpus = len(cards)
    allocated_gpus = alloc.get("gpu", 0)
    available_gpus = total_gpus - allocated_gpus

    if alloc.get("node", 0) == 0:
        status = "IDLE"
    elif available_cpus == 0 and (total_gpus == 0 or available_gpus == 0):
        status = "FULLY OCCUPIED"
    else:
        status = "PARTIAL"

    lines = [f"NODE: {name} [{status}]"]
    lines.append(f"  CPUs:         {available_cpus}/{total_cpus} available")
    lines.append(f"  CPU model:    {node.get('cpu_model', 'Unknown')}")
    lines.append(f"  Architecture: {node.get('architecture', 'Unknown')}")
    lines.append(f"  Memory:       {available_mem_gb}/{memory_gb} GB available")
    lines.append(f"  Topology:     {sockets} socket(s) × {cores_per_socket} cores × {threads_per_core} threads")

    if cards:
        gpu_model = cards[0].get("model", "Unknown")
        lines.append(f"  GPUs:         {available_gpus}/{total_gpus} available ({gpu_model})")

    lines.append(f"  Partitions:   [{', '.join(node.get('partitions', []))}]")
    lines.append(f"  OS:           {node.get('os_name', '')} {node.get('os_release', '')}")

    return "\n".join(lines)
