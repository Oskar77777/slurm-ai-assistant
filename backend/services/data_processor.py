"""
Data processor for eX3 cluster API responses.
Summarizes raw node data into compact format with calculated availability.
"""

from typing import Any


def summarize_nodes(nodes_data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert raw node data to structured summary grouped by availability status.

    Args:
        nodes_data: Raw node data from eX3 API (dict with node names as keys)

    Returns:
        Dict with summary stats and nodes grouped by status
    """
    nodes_by_status = {"idle": [], "partial": [], "full": []}

    for name, node in nodes_data.items():
        # Skip non-compute nodes (login nodes, admin nodes, etc.)
        partitions = node.get("partitions", [])
        if not partitions:
            continue

        # Calculate total CPUs
        sockets = node.get("sockets", 1)
        cores_per_socket = node.get("cores_per_socket", 1)
        threads_per_core = node.get("threads_per_core", 1)
        total_cpus = sockets * cores_per_socket * threads_per_core

        # Calculate total GPUs
        cards = node.get("cards", [])
        total_gpus = len(cards)

        # Get allocation info
        alloc = node.get("alloc_tres", {})
        allocated_cpus = alloc.get("cpu", 0)
        allocated_gpus = alloc.get("gpu", 0)

        # Calculate available resources
        available_cpus = total_cpus - allocated_cpus
        available_gpus = total_gpus - allocated_gpus

        # Determine status
        if alloc.get("node", 0) == 0:
            status = "idle"
        elif available_cpus == 0 and (total_gpus == 0 or available_gpus == 0):
            status = "full"
        else:
            status = "partial"

        # Get GPU model if available (shortened)
        gpu_model = None
        if cards:
            model = cards[0].get("model", "")
            # Shorten common prefixes
            gpu_model = model.replace("NVIDIA ", "")

        # Calculate memory in GB
        memory_kb = node.get("memory", 0)
        memory_gb = round(memory_kb / 1024 / 1024, 0)

        node_info = {
            "name": name,
            "type": "GPU" if total_gpus > 0 else "CPU",
            "cpus": f"{available_cpus}/{total_cpus}",
            "mem_gb": int(memory_gb),
            "partitions": partitions
        }

        # Only include GPU info for GPU nodes
        if total_gpus > 0:
            node_info["gpus"] = f"{available_gpus}/{total_gpus}"
            node_info["gpu_model"] = gpu_model

        nodes_by_status[status].append(node_info)

    # Sort nodes within each group by name
    for status in nodes_by_status:
        nodes_by_status[status].sort(key=lambda x: x["name"])

    # Build summary
    total_nodes = sum(len(nodes) for nodes in nodes_by_status.values())
    gpu_nodes_idle = [n for n in nodes_by_status["idle"] if n["type"] == "GPU"]
    cpu_nodes_idle = [n for n in nodes_by_status["idle"] if n["type"] == "CPU"]

    # Format as human-readable text so LLM can present it directly
    lines = []
    lines.append(f"CLUSTER STATUS: {total_nodes} compute nodes total")
    lines.append(f"  - {len(nodes_by_status['idle'])} idle ({len(gpu_nodes_idle)} GPU, {len(cpu_nodes_idle)} CPU)")
    lines.append(f"  - {len(nodes_by_status['partial'])} partially used")
    lines.append(f"  - {len(nodes_by_status['full'])} fully occupied")
    lines.append("")

    if nodes_by_status["idle"]:
        lines.append(f"IDLE NODES ({len(nodes_by_status['idle'])}):")
        for n in nodes_by_status["idle"]:
            if n["type"] == "GPU":
                lines.append(f"  {n['name']}: {n['cpus']} CPUs, {n['gpus']} {n['gpu_model']} GPUs, {n['mem_gb']} GB RAM [{', '.join(n['partitions'])}]")
            else:
                lines.append(f"  {n['name']}: {n['cpus']} CPUs, {n['mem_gb']} GB RAM [{', '.join(n['partitions'])}]")

    if nodes_by_status["partial"]:
        lines.append("")
        lines.append(f"PARTIAL NODES ({len(nodes_by_status['partial'])}):")
        for n in nodes_by_status["partial"]:
            if n["type"] == "GPU":
                lines.append(f"  {n['name']}: {n['cpus']} CPUs, {n['gpus']} {n['gpu_model']} GPUs, {n['mem_gb']} GB RAM [{', '.join(n['partitions'])}]")
            else:
                lines.append(f"  {n['name']}: {n['cpus']} CPUs, {n['mem_gb']} GB RAM [{', '.join(n['partitions'])}]")

    if nodes_by_status["full"]:
        lines.append("")
        lines.append(f"FULLY OCCUPIED NODES ({len(nodes_by_status['full'])}):")
        for n in nodes_by_status["full"]:
            lines.append(f"  {n['name']}: [{', '.join(n['partitions'])}]")

    return "\n".join(lines)
