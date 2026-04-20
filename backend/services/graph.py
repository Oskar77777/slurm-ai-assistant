import re
import json
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

import config
from services.ollama_client import ollama_client, detect_prompt_intent, detect_fetch_intent, build_system_prompt
from services.ex3_client import ex3_client
from services.data_processor import summarize_nodes, summarize_gpu_nodes, summarize_cpu_nodes, format_single_node, summarize_partitions
from services.slurm_validator import validate_script, format_errors_for_llm
from services.resource_planner import recommend_gpu_allocation, extract_gpu_count, detect_node_query_intent

logger = logging.getLogger(__name__)

_FETCH_RE = re.compile(r"\[FETCH:\s*([^\]]+)\]")
_NODE_RE = re.compile(r"\b((?:gh|g|n|v)\d{3,4})\b", re.IGNORECASE)
_SBATCH_RE = re.compile(r"#SBATCH\s+--(\S+?)(?:=(.*))?$")

# Intents that always need cluster data — fetched deterministically before the LLM call
_INTENT_FETCH_MAP = {
    "node_query":      "nodes_list",
    "script_write":    "nodes_list",
    "script_adapt":    "nodes_list",
    "job_query":       "jobs_list",
    "partition_query": "partitions_list",
}

_DIV  = "=" * 60
_SEP  = "-" * 60


# ── State ─────────────────────────────────────────────────────────────────────

class AssistantState(TypedDict):
    messages: list[dict]
    system_prompt: str
    original_directives: dict[str, str]
    fetch_count: int
    prefetched_tool: str  # tool pre-fetched in preprocess, or "" if none
    llm_response: str
    response: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_fetch_marker(text: str) -> str | None:
    match = _FETCH_RE.search(text)
    return match.group(1).strip() if match else None


def _extract_node_name(messages: list[dict]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if content.startswith("[SYSTEM DATA"):
            continue
        match = _NODE_RE.search(content)
        return match.group(1).lower() if match else None
    return None


def _extract_sbatch_directives(script: str) -> dict[str, str]:
    directives = {}
    for line in script.splitlines():
        match = _SBATCH_RE.match(line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    return directives


def _extract_user_script(messages: list[dict]) -> str | None:
    for msg in messages:
        if msg.get("role") == "user" and "#SBATCH" in msg.get("content", ""):
            return msg["content"]
    return None


def _restore_missing_directives(response: str, original_directives: dict[str, str]) -> str:
    block_match = re.search(r"(```bash\n)(.*?)(```)", response, re.DOTALL)
    if not block_match:
        return response

    block_content = block_match.group(2)
    missing = {
        k: v for k, v in original_directives.items()
        if not re.search(rf"#SBATCH\s+--{re.escape(k)}", block_content)
    }
    if not missing:
        return response

    inject_lines = "\n".join(
        f"#SBATCH --{k}={v}" if v else f"#SBATCH --{k}" for k, v in missing.items()
    )
    logger.info(f"Restoring {len(missing)} missing directive(s): {list(missing.keys())}")

    lines = block_content.splitlines()
    last_sbatch = max(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("#SBATCH")),
        default=None,
    )
    if last_sbatch is not None:
        lines.insert(last_sbatch + 1, inject_lines)
        new_block = "\n".join(lines)
    else:
        new_block = inject_lines + "\n" + block_content

    return response[: block_match.start(2)] + new_block + response[block_match.end(2):]


def _is_first_turn(messages: list[dict]) -> bool:
    """True if there are no real assistant responses yet in the conversation history.
    Injected 'I've fetched...' messages don't count as real turns."""
    return not any(
        m["role"] == "assistant"
        and not m["content"].startswith("I've fetched")
        and not m["content"].startswith("I tried to fetch")
        for m in messages
    )


def _log_messages(messages: list[dict], system_prompt: str) -> None:
    """Log the full message list being sent to Ollama."""
    logger.info(_SEP)
    logger.info("MESSAGES SENT TO LLM:")
    logger.info(f"  [system] ({len(system_prompt)} chars):\n{system_prompt}")
    logger.info(_SEP)
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg["content"]
        logger.info(f"  [{role}] (msg {i + 1}, {len(content)} chars):\n{content}")
    logger.info(_SEP)


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def preprocess_node(state: AssistantState) -> dict:
    """Validate SLURM content, detect intent, deterministically pre-fetch cluster data."""
    latest_user_msg = next(
        (m["content"] for m in reversed(state["messages"])
         if m["role"] == "user" and not m["content"].startswith("[SYSTEM DATA")),
        "(none)"
    )
    logger.info(_DIV)
    logger.info("NEW REQUEST")
    logger.info(f"  User: {latest_user_msg}")
    logger.info(_DIV)

    messages = []
    for msg in state["messages"]:
        content = msg["content"]
        if msg["role"] == "user" and "#SBATCH" in content:
            errors = validate_script(content)
            if errors:
                content = f"{content}\n\n{format_errors_for_llm(errors)}"
                logger.info(f"SLURM validation: {len(errors)} error(s) found and appended")
        messages.append({"role": msg["role"], "content": content})

    user_script = _extract_user_script(messages)
    original_directives = _extract_sbatch_directives(user_script) if user_script else {}

    intent = detect_prompt_intent(messages)       # lookback — for system prompt
    fetch_intent = detect_fetch_intent(messages)  # latest message only — for pre-fetch
    first_turn = _is_first_turn(messages)
    system_prompt = build_system_prompt(intent, first_turn=first_turn)
    logger.info(f"Intent: {intent} | Fetch intent: {fetch_intent} | Turn: {'first' if first_turn else 'follow-up'} | System prompt: {len(system_prompt)} chars")

    # ── Specific node pre-fetch (user named a node explicitly) ────────────────
    specific_node = _extract_node_name(messages)
    if specific_node:
        try:
            logger.info(f"Pre-fetching specific node: {specific_node}")
            node_data = await ex3_client.get_node_info(specific_node)
            if not node_data:
                messages.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM DATA] Node '{specific_node}' was not found in the eX3 cluster. "
                        f"Tell the user that this node is currently not available or does not exist, "
                        f"and ask if they would like to use a different node."
                    ),
                })
                logger.warning(f"Node {specific_node} not found in cluster")
            else:
                node_str = format_single_node(node_data)
                messages.append({
                    "role": "assistant",
                    "content": f"I've fetched the details for node {specific_node}.",
                })
                node_content = f"[SYSTEM DATA - Real-time info for node {specific_node}]\n\n{node_str}"
                if intent in ("script_write", "script_adapt"):
                    node_content += (
                        f"\n\nUse ONLY the partitions listed above for this node. "
                        f"Set --mem to a reasonable amount for the job (e.g. 32G), not the full node memory."
                    )
                messages.append({"role": "user", "content": node_content})
                logger.info(f"Pre-fetched node data for {specific_node}:\n{node_str}")
        except Exception as e:
            logger.warning(f"Could not pre-fetch node {specific_node}: {e}")

    # ── Deterministic fetch based on latest message intent only ──────────────
    elif fetch_intent in _INTENT_FETCH_MAP:
        tool = _INTENT_FETCH_MAP[fetch_intent]
        try:
            logger.info(f"Deterministic pre-fetch: {tool} (intent={intent})")
            cluster_data = await ex3_client.call_by_tool_name(tool)

            if tool in ["nodes_list", "nodes_info"]:
                node_intent = detect_node_query_intent(messages)
                if node_intent == "gpu":
                    data_str = summarize_gpu_nodes(cluster_data)
                    num_gpus = extract_gpu_count(messages) or 1
                    recommendation = recommend_gpu_allocation(cluster_data, num_gpus)
                    data_str = data_str + "\n\n" + recommendation
                    logger.info(f"Added GPU recommendation for {num_gpus} GPUs")
                elif node_intent == "cpu":
                    data_str = summarize_cpu_nodes(cluster_data)
                else:
                    data_str = summarize_nodes(cluster_data)
            elif tool == "partitions_list":
                data_str = summarize_partitions(cluster_data)
            else:
                data_str = json.dumps(cluster_data, indent=2)

            logger.info(f"Pre-fetched data ({len(data_str)} chars):\n{data_str}")
            messages.append({
                "role": "assistant",
                "content": "I've fetched the cluster data.",
            })
            messages.append({
                "role": "user",
                "content": f"[SYSTEM DATA - Real-time data from eX3 cluster API]\n\n{data_str}",
            })
        except Exception as e:
            logger.warning(f"Deterministic pre-fetch failed for '{tool}': {e}")

    prefetched_tool = ""
    if specific_node:
        prefetched_tool = f"node_info:{specific_node}"
    elif fetch_intent in _INTENT_FETCH_MAP:
        prefetched_tool = _INTENT_FETCH_MAP[fetch_intent]

    return {
        "messages": messages,
        "system_prompt": system_prompt,
        "original_directives": original_directives,
        "fetch_count": 0,
        "prefetched_tool": prefetched_tool,
        "llm_response": "",
        "response": "",
    }


async def call_llm_node(state: AssistantState) -> dict:
    """Call Ollama and store the raw response."""
    iteration = state["fetch_count"] + 1
    logger.info(f"LLM CALL — iteration {iteration}")
    _log_messages(state["messages"], state["system_prompt"])

    response = await ollama_client.chat(state["messages"], system_prompt=state["system_prompt"])

    logger.info(f"LLM RESPONSE (iteration {iteration}):\n{response}")
    return {"llm_response": response}


async def fetch_data_node(state: AssistantState) -> dict:
    """Call eX3 API based on the fetch marker and inject the result into messages."""
    tool_name = _parse_fetch_marker(state["llm_response"])
    messages = list(state["messages"])

    logger.info(_SEP)
    logger.info(f"FETCH REQUESTED: {tool_name}")

    try:
        cluster_data = await ex3_client.call_by_tool_name(tool_name)

        if tool_name in ["nodes_list", "nodes_info"]:
            intent = detect_node_query_intent(messages)
            if intent == "gpu":
                data_str = summarize_gpu_nodes(cluster_data)
                num_gpus = extract_gpu_count(messages) or 1
                recommendation = recommend_gpu_allocation(cluster_data, num_gpus)
                data_str = data_str + "\n\n" + recommendation
                logger.info(f"GPU intent: injected resource recommendation for {num_gpus} GPUs")
            elif intent == "cpu":
                data_str = summarize_cpu_nodes(cluster_data)
            else:
                data_str = summarize_nodes(cluster_data)
            logger.info(f"Node query intent: {intent}")
        elif tool_name.startswith("node_info:"):
            data_str = format_single_node(cluster_data)
        elif tool_name == "partitions_list":
            data_str = summarize_partitions(cluster_data)
        else:
            data_str = json.dumps(cluster_data, indent=2)

        logger.info(f"FORMATTED DATA INJECTED INTO LLM ({len(data_str)} chars):\n{data_str}")
        logger.info(_SEP)

        messages.append({
            "role": "assistant",
            "content": "I've fetched the cluster data. Let me analyze it for you.",
        })
        messages.append({
            "role": "user",
            "content": (
                f"[SYSTEM DATA - This is real-time data from the eX3 cluster API, "
                f"use it to answer my question]\n\n{data_str}"
            ),
        })
    except Exception as e:
        logger.error(f"eX3 API error for tool '{tool_name}': {e}")
        messages.append({
            "role": "assistant",
            "content": f"I tried to fetch cluster data but encountered an error: {e}",
        })
        messages.append({
            "role": "user",
            "content": "That's okay, please provide what general help you can without the real-time data.",
        })

    return {"messages": messages, "fetch_count": state["fetch_count"] + 1}


async def postprocess_node(state: AssistantState) -> dict:
    """Restore missing #SBATCH directives and set the final response."""
    response = state["llm_response"]
    if state["original_directives"]:
        response = _restore_missing_directives(response, state["original_directives"])
        if response != state["llm_response"]:
            logger.info(f"FINAL RESPONSE (after directive restoration):\n{response}")
    logger.info(_DIV)
    return {"response": response}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_llm(state: AssistantState) -> str:
    fetch_tool = _parse_fetch_marker(state["llm_response"])
    if fetch_tool and state["fetch_count"] < config.MAX_FETCH_ITERATIONS:
        if fetch_tool == state.get("prefetched_tool", ""):
            logger.info(f"Skipping duplicate fetch for already pre-fetched tool: {fetch_tool}")
            return "postprocess"
        return "fetch_data"
    return "postprocess"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AssistantState)

    graph.add_node("preprocess", preprocess_node)
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("postprocess", postprocess_node)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "call_llm")
    graph.add_conditional_edges(
        "call_llm",
        route_after_llm,
        {"fetch_data": "fetch_data", "postprocess": "postprocess"},
    )
    graph.add_edge("fetch_data", "call_llm")
    graph.add_edge("postprocess", END)

    return graph.compile()


assistant_graph = build_graph()
