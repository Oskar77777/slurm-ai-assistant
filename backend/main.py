import re
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from models import ChatRequest, ChatResponse
from services.ollama_client import ollama_client, detect_prompt_intent, build_system_prompt
from services.ex3_client import ex3_client
from services.data_processor import summarize_nodes, summarize_gpu_nodes, summarize_cpu_nodes, format_single_node
from services.slurm_validator import validate_script, format_errors_for_llm
from services.resource_planner import recommend_gpu_allocation, extract_gpu_count, detect_node_query_intent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="eX3 Cluster AI Assistant")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_fetch_marker(response: str) -> str | None:
    """Extract tool name from [FETCH: tool_name] marker in response."""
    match = re.search(r"\[FETCH:\s*([^\]]+)\]", response)
    if match:
        return match.group(1).strip()
    return None


def extract_node_name(messages: list[dict]) -> str | None:
    """Extract a specific node name from the latest user message only."""
    pattern = re.compile(r"\b((?:gh|g|n|v)\d{3,4})\b", re.IGNORECASE)
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if content.startswith("[SYSTEM DATA"):
            continue
        # Only check the single most recent real user message
        match = pattern.search(content)
        return match.group(1).lower() if match else None
    return None


def preprocess_message(content: str) -> str:
    """Check for SLURM scripts and validate them."""
    if "#SBATCH" in content:
        errors = validate_script(content)
        if errors:
            validation_msg = format_errors_for_llm(errors)
            return f"{content}\n\n{validation_msg}"
    return content


def extract_sbatch_directives(script: str) -> dict[str, str]:
    """Extract #SBATCH directives from a script as {flag: value} dict."""
    directives = {}
    for line in script.splitlines():
        line = line.strip()
        match = re.match(r"#SBATCH\s+--(\S+?)(?:=(.*))?$", line)
        if match:
            key = match.group(1)
            value = match.group(2) or ""
            directives[key] = value.strip()
    return directives


def extract_user_script(messages: list[dict]) -> str | None:
    """Return the first user message that contains a SLURM script, or None."""
    for msg in messages:
        if msg.get("role") == "user" and "#SBATCH" in msg.get("content", ""):
            return msg["content"]
    return None


def restore_missing_directives(response: str, original_directives: dict[str, str]) -> str:
    """
    Find the bash code block in the LLM response and inject any directives
    from the original script that are missing from it.
    """
    block_match = re.search(r"(```bash\n)(.*?)(```)", response, re.DOTALL)
    if not block_match:
        return response

    block_content = block_match.group(2)

    # Find which directives are missing from the LLM's output script
    missing = {}
    for key, value in original_directives.items():
        pattern = rf"#SBATCH\s+--{re.escape(key)}"
        if not re.search(pattern, block_content):
            missing[key] = value

    if not missing:
        return response

    # Build lines to inject and insert them after the last existing #SBATCH line
    inject_lines = "\n".join(
        f"#SBATCH --{k}={v}" if v else f"#SBATCH --{k}" for k, v in missing.items()
    )
    logger.info(f"Restoring {len(missing)} missing directive(s): {list(missing.keys())}")

    # Insert after the last #SBATCH line in the block
    lines = block_content.splitlines()
    last_sbatch = max((i for i, l in enumerate(lines) if l.strip().startswith("#SBATCH")), default=None)
    if last_sbatch is not None:
        lines.insert(last_sbatch + 1, inject_lines)
        new_block = "\n".join(lines)
    else:
        new_block = inject_lines + "\n" + block_content

    return response[:block_match.start(2)] + new_block + response[block_match.end(2):]


@app.on_event("startup")
async def startup_event():
    """Log configuration on startup."""
    logger.info(f"EX3_API_BASE_URL: {config.EX3_API_BASE_URL}")
    logger.info(f"EX3_DEFAULT_CLUSTER: {config.EX3_DEFAULT_CLUSTER}")
    logger.info(f"OLLAMA_BASE_URL: {config.OLLAMA_BASE_URL}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with intelligent eX3 API calling."""
    # Preprocess user messages to validate SLURM scripts
    messages = []
    for msg in request.messages:
        content = msg.content
        if msg.role == "user":
            content = preprocess_message(content)
        messages.append({"role": msg.role, "content": content})

    # Extract directives from any user-submitted script for post-processing
    user_script = extract_user_script(messages)
    original_directives = extract_sbatch_directives(user_script) if user_script else {}

    # Build a focused system prompt based on what the user is asking
    prompt_intent = detect_prompt_intent(messages)
    system_prompt = build_system_prompt(prompt_intent)
    logger.info(f"Prompt intent: {prompt_intent}")

    # If the user mentioned a specific node, pre-fetch its data deterministically
    specific_node = extract_node_name(messages)
    if specific_node:
        try:
            logger.info(f"Pre-fetching node info for: {specific_node}")
            node_data = await ex3_client.get_node_info(specific_node)
            if not node_data:
                # Node not found in the cluster — tell the LLM so it can inform the user
                messages.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM DATA] Node '{specific_node}' was not found in the eX3 cluster. "
                        f"Tell the user that this node is currently not available or does not exist, "
                        f"and ask if they would like to use a different node."
                    )
                })
                logger.warning(f"Node {specific_node} returned empty response from API")
            else:
                node_str = format_single_node(node_data)
                messages.append({
                    "role": "assistant",
                    "content": f"I've fetched the details for node {specific_node}."
                })
                messages.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM DATA - Real-time info for node {specific_node}]\n\n{node_str}\n\n"
                        f"Use ONLY the partitions listed above for this node. "
                        f"Set --mem to a reasonable amount for the job (e.g. 32G), not the full node memory."
                    )
                })
                logger.info(f"Injected node data for {specific_node}")
        except Exception as e:
            logger.warning(f"Could not pre-fetch node {specific_node}: {e}")

    iterations = 0

    while iterations < config.MAX_FETCH_ITERATIONS:
        iterations += 1
        logger.info(f"Iteration {iterations}")

        try:
            response = await ollama_client.chat(messages, system_prompt=system_prompt)
            logger.info(f"LLM response: {response[:200]}...")
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            raise HTTPException(status_code=503, detail=f"LLM service error: {str(e)}")

        tool_name = parse_fetch_marker(response)
        logger.info(f"Parsed tool name: {tool_name}")

        if tool_name:
            # LLM wants to fetch cluster data
            try:
                logger.info(f"Fetching data for tool: {tool_name}")
                cluster_data = await ex3_client.call_by_tool_name(tool_name)

                # Summarize node data to human-readable format
                if tool_name in ["nodes_list", "nodes_info"]:
                    intent = detect_node_query_intent(messages)
                    if intent == "gpu":
                        data_str = summarize_gpu_nodes(cluster_data)
                    elif intent == "cpu":
                        data_str = summarize_cpu_nodes(cluster_data)
                    else:
                        data_str = summarize_nodes(cluster_data)
                    logger.info(f"Node query intent: {intent}")

                    if intent == "gpu":
                        num_gpus = extract_gpu_count(messages) or 1
                        recommendation = recommend_gpu_allocation(cluster_data, num_gpus)
                        data_str = data_str + "\n\n" + recommendation
                        logger.info(f"Injected resource recommendation for {num_gpus} GPUs")
                    logger.info(f"Summarized node data ({len(data_str)} chars)")
                elif tool_name.startswith("node_info:"):
                    data_str = format_single_node(cluster_data)
                else:
                    data_str = json.dumps(cluster_data, indent=2)
                logger.info(f"Got cluster data ({len(data_str)} chars): {data_str[:200]}...")

                # Add data as assistant message showing successful fetch, then continue
                messages.append({
                    "role": "assistant",
                    "content": f"I've fetched the cluster data. Let me analyze it for you."
                })
                messages.append({
                    "role": "user",
                    "content": f"[SYSTEM DATA - This is real-time data from the eX3 cluster API, use it to answer my question]\n\n{data_str}"
                })
                continue
            except Exception as e:
                logger.error(f"eX3 API error: {str(e)}")
                messages.append({
                    "role": "assistant",
                    "content": f"I tried to fetch cluster data but encountered an error: {str(e)}"
                })
                messages.append({
                    "role": "user",
                    "content": "That's okay, please provide what general help you can without the real-time data."
                })
                continue
        else:
            # No fetch marker, return the response
            logger.info("No fetch marker, returning response")
            if original_directives:
                response = restore_missing_directives(response, original_directives)
            return ChatResponse(response=response)

    # Max iterations reached
    logger.warning("Max iterations reached")
    return ChatResponse(
        response="I apologize, but I encountered an issue processing your request. Please try asking your question again."
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.BACKEND_HOST, port=config.BACKEND_PORT)
