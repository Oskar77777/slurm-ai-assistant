# Backend configuration
import os

# Server settings
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000

# eX3 Cluster API settings
EX3_API_BASE_URL = os.getenv("EX3_API_BASE_URL", "https://localhost:12200/api/v2")
EX3_DEFAULT_CLUSTER = "ex3.simula.no"
EX3_API_TIMEOUT = 30.0

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = 120.0

# Chat settings
MAX_FETCH_ITERATIONS = 5  # Prevent infinite fetch loops
