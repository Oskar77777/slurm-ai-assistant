import re
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from models import ChatRequest, ChatResponse
from services.ollama_client import ollama_client
from services.ex3_client import ex3_client
from services.data_processor import summarize_nodes
from services.slurm_validator import validate_script, format_errors_for_llm
from services.resource_planner import recommend_gpu_allocation, extract_gpu_count

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


def preprocess_message(content: str) -> str:
    """Check for SLURM scripts and validate them."""
    if "#SBATCH" in content:
        errors = validate_script(content)
        if errors:
            validation_msg = format_errors_for_llm(errors)
            return f"{content}\n\n{validation_msg}"
    return content


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
    iterations = 0

    while iterations < config.MAX_FETCH_ITERATIONS:
        iterations += 1
        logger.info(f"Iteration {iterations}")

        try:
            response = await ollama_client.chat(messages)
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
                    data_str = summarize_nodes(cluster_data)
                    num_gpus = extract_gpu_count(messages)
                    if num_gpus:
                        recommendation = recommend_gpu_allocation(cluster_data, num_gpus)
                        data_str = data_str + "\n\n" + recommendation
                        logger.info(f"Injected resource recommendation for {num_gpus} GPUs")
                    logger.info(f"Summarized node data ({len(data_str)} chars)")
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
            return ChatResponse(response=response)

    # Max iterations reached
    logger.warning("Max iterations reached")
    return ChatResponse(
        response="I apologize, but I encountered an issue processing your request. Please try asking your question again."
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.BACKEND_HOST, port=config.BACKEND_PORT)
