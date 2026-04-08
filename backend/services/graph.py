import re
import json
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

import config
from services.ollama_client import ollama_client, detect_prompt_intent, build_system_prompt
from services.ex3_client import ex3_client
from services.data_processor import summarize_nodes, summarize_gpu_nodes, summarize_cpu_nodes, format_single_node
from services.slurm_validator import validate_script, format_errors_for_llm
from services.resource_planner import recommend_gpu_allocation, extract_gpu_count, detect_node_query_intent

logger = logging.getLogger(__name__)

_FETCH_RE = re.compile(r"\[FETCH:\s*([^\]]+)\]")
_NODE_RE = re.compile(r"\b((?:gh|g|n|v)\d{3,4})\b", re.IGNORECASE)
_SBATCH_RE = re.compile(r"#SBATCH\s+--(\S+?)(?:=(.*))?$")


# ── State ─────────────────────────────────────────────────────────────────────

class AssistantState(TypedDict):
    messages: list[dict]
    system_prompt: str
    original_directives: dict[str, str]
    fetch_count: int
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


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def preprocess_node(state: AssistantState) -> dict:
    """Validate SLURM content, detect intent, pre-fetch specific node if mentioned."""
    messages = []
    for msg in state["messages"]:
        content = msg["content"]
        if msg["role"] == "user" and "#SBATCH" in content:
            errors = validate_script(content)
            if errors:
                content = f"{content}\n\n{format_errors_for_llm(errors)}"
        messages.append({"role": msg["role"], "content": content})

    user_script = _extract_user_script(messages)
    original_directives = _extract_sbatch_directives(user_script) if user_script else {}

    intent = detect_prompt_intent(messages)
    system_prompt = build_system_prompt(intent)
    logger.info(f"Prompt intent: {intent}")

    specific_node = _extract_node_name(messages)
    if specific_node:
        try:
            logger.info(f"Pre-fetching node info for: {specific_node}")
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
                logger.warning(f"Node {specific_node} returned empty response from API")
            else:
                node_str = format_single_node(node_data)
                messages.append({
                    "role": "assistant",
                    "content": f"I've fetched the details for node {specific_node}.",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM DATA - Real-time info for node {specific_node}]\n\n{node_str}\n\n"
                        f"Use ONLY the partitions listed above for this node. "
                        f"Set --mem to a reasonable amount for the job (e.g. 32G), not the full node memory."
                    ),
                })
                logger.info(f"Injected node data for {specific_node}")
        except Exception as e:
            logger.warning(f"Could not pre-fetch node {specific_node}: {e}")

    return {
        "messages": messages,
        "system_prompt": system_prompt,
        "original_directives": original_directives,
        "fetch_count": 0,
        "llm_response": "",
        "response": "",
    }


async def call_llm_node(state: AssistantState) -> dict:
    """Call Ollama and store the raw response."""
    logger.info(f"LLM call (fetch_count={state['fetch_count']})")
    response = await ollama_client.chat(state["messages"], system_prompt=state["system_prompt"])
    logger.info(f"LLM response: {response[:200]}...")
    return {"llm_response": response}


async def fetch_data_node(state: AssistantState) -> dict:
    """Call eX3 API based on the fetch marker and inject the result into messages."""
    tool_name = _parse_fetch_marker(state["llm_response"])
    messages = list(state["messages"])

    try:
        logger.info(f"Fetching data for tool: {tool_name}")
        cluster_data = await ex3_client.call_by_tool_name(tool_name)

        if tool_name in ["nodes_list", "nodes_info"]:
            intent = detect_node_query_intent(messages)
            if intent == "gpu":
                data_str = summarize_gpu_nodes(cluster_data)
                num_gpus = extract_gpu_count(messages) or 1
                recommendation = recommend_gpu_allocation(cluster_data, num_gpus)
                data_str = data_str + "\n\n" + recommendation
                logger.info(f"Injected resource recommendation for {num_gpus} GPUs")
            elif intent == "cpu":
                data_str = summarize_cpu_nodes(cluster_data)
            else:
                data_str = summarize_nodes(cluster_data)
            logger.info(f"Node query intent: {intent}, summarized {len(data_str)} chars")
        elif tool_name.startswith("node_info:"):
            data_str = format_single_node(cluster_data)
        else:
            data_str = json.dumps(cluster_data, indent=2)

        logger.info(f"Got cluster data ({len(data_str)} chars): {data_str[:200]}...")
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
        logger.error(f"eX3 API error: {e}")
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
    return {"response": response}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_llm(state: AssistantState) -> str:
    if _parse_fetch_marker(state["llm_response"]) and state["fetch_count"] < config.MAX_FETCH_ITERATIONS:
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
