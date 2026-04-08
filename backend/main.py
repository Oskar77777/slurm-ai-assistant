import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from models import ChatRequest, ChatResponse
from services.graph import assistant_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="eX3 Cluster AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info(f"EX3_API_BASE_URL: {config.EX3_API_BASE_URL}")
    logger.info(f"EX3_DEFAULT_CLUSTER: {config.EX3_DEFAULT_CLUSTER}")
    logger.info(f"OLLAMA_BASE_URL: {config.OLLAMA_BASE_URL}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    try:
        result = await assistant_graph.ainvoke({
            "messages": messages,
            "system_prompt": "",
            "original_directives": {},
            "fetch_count": 0,
            "llm_response": "",
            "response": "",
        })
    except Exception as e:
        logger.error(f"Graph error: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    response = result.get("response") or (
        "I apologize, but I encountered an issue processing your request. "
        "Please try asking your question again."
    )
    return ChatResponse(response=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.BACKEND_HOST, port=config.BACKEND_PORT)
