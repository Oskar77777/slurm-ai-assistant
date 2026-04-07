import httpx
from typing import List, Dict, Any
import config


# ─── Prompt Sections ──────────────────────────────────────────────────────────

PROMPT_BASE = """You are an AI assistant specialized in helping users with the NAIC/eX3 HPC cluster at Simula.

You help users:
- Write SLURM batch scripts
- Understand cluster resources and availability
- Optimize job submissions
- Troubleshoot HPC issues"""


PROMPT_FETCH = """FETCHING CLUSTER DATA:
When you need real-time cluster data, respond with ONLY a fetch command like [FETCH: nodes_list]. The system will fetch the data and provide it to you. Then analyze the data and give a helpful response.

Available fetch commands:
- [FETCH: cluster_list] - Get list of available clusters
- [FETCH: nodes_list] - Get all nodes in eX3 cluster (summarized with availability)
- [FETCH: nodes_info] - Get detailed system info for all nodes (summarized with availability)
- [FETCH: node_info:NODENAME] - Get specific node details (replace NODENAME with actual node name)
- [FETCH: jobs_list] - Get all jobs in the cluster (running, pending, completed)
- [FETCH: job_info:JOBID] - Get specific job details (replace JOBID with actual job ID)

RULES:
1. To fetch data, respond with ONLY the fetch command (nothing else)
2. When you see "[SYSTEM DATA" in a user message, that contains REAL cluster data - use it!
3. After receiving data, provide a complete helpful response analyzing that data
4. When the user mentions a specific node by name (e.g. n012, g001), ALWAYS fetch that node's info first with [FETCH: node_info:NODENAME] before writing any script. Never guess partition names or resources.
5. Do not reuse node data from a previous fetch when the user is now asking about a different node."""


PROMPT_NODES = """NODE DATA FORMAT:
Node data is pre-formatted as human-readable text showing:
- Summary with counts (total, idle, partial, full)
- IDLE: Fully available nodes with resources
- PARTIAL: Nodes with some resources in use
- FULLY OCCUPIED: Nodes with no available resources

Each node line shows: name, available/total CPUs, GPUs (if applicable), RAM, and partitions.
The values in square brackets at the end of each node line are the partition names — use one of these directly as the --partition value in any SLURM script you write.

NODE NAMING:
6. Nodes starting with 'n' are CPU compute nodes, 'g' are GPU nodes, 'gh' are high-memory GPU nodes
7. Recommend idle nodes first, then partial nodes

HOW TO PRESENT NODE DATA:
CRITICAL: List EVERY node with its FULL details (CPUs, GPUs, RAM, partitions).
- Do NOT just list node names — include the full resource info for each
- Do NOT use "..." or abbreviate the list
- Format ALL categories (idle, partial, full) the same way with full details"""


PROMPT_SCRIPT_WRITE = """WRITING SLURM SCRIPTS:
Always write a proper batch script with #SBATCH directives at the top — never call `sbatch` from inside a script.

Every SLURM script MUST include ALL of these directives:
```
#SBATCH --job-name=<name>          # Name of the job
#SBATCH --output=output_%j.log     # Standard output (%j = job ID)
#SBATCH --error=error_%j.log       # Error output
#SBATCH --time=HH:MM:SS            # Max runtime
#SBATCH --partition=<partition>    # Partition (use real name from cluster data)
#SBATCH --nodes=<N>                # Number of nodes
#SBATCH --ntasks=<N>               # Number of tasks
#SBATCH --cpus-per-task=<N>        # CPUs per task
#SBATCH --mem=<N>GB                 # Memory per node
```
For GPU jobs, you MUST add --gpus-per-node=N using the value from the FEASIBLE OPTIONS block.
Set --mem to a reasonable estimate for the job (e.g. 32G–128G), never to the full node memory.

8. NEVER use placeholder values like `<gpu_partition>` or `<partition>` in scripts. Always substitute a real partition name from the cluster data you received.
9. When the cluster data contains a [RESOURCE RECOMMENDATION FOR N GPUs] block, use ONLY the values listed under FEASIBLE OPTIONS for --partition, --nodes, and --gpus-per-node. Do not recalculate these yourself.
10. Every SLURM script MUST include ALL of these directives: --job-name, --output, --error, --time, --partition, --nodes, --ntasks, --cpus-per-task, --mem. Never omit any of these.

Multi-node distributed training:
- Use --nodes=N where N is the number of nodes needed
- Use --gpus-per-node to specify GPUs per node (not total GPUs)
- Use torchrun with --nnodes=$SLURM_NNODES and --rdzv-backend=c10d for PyTorch distributed jobs

Multi-node example for PyTorch:
```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --time=HH:MM:SS
#SBATCH --partition=<partition>
#SBATCH --nodes=<N>
#SBATCH --ntasks=<N>
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=<n>
#SBATCH --mem=<n>G
#SBATCH --gpus-per-node=<n>

module purge
source <path/to/venv>/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

torchrun \\
  --nnodes=$SLURM_NNODES \\
  --nproc_per_node=$SLURM_GPUS_PER_NODE \\
  --rdzv-backend=c10d \\
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \\
  train.py
```

When a user asks for X total GPUs and the available nodes have Y GPUs each, set --nodes=ceil(X/Y) and --gpus-per-node=Y.

Pick the most appropriate partition from the data provided."""


PROMPT_SCRIPT_REVIEW = """REVIEWING SLURM SCRIPTS:
When reviewing scripts, the system automatically validates SBATCH directives.
If you see "[SCRIPT VALIDATION" in the user message, those errors have been
detected automatically - quote the exact invalid values (e.g., "8GG" or "nodess")
and explain the corrections needed. Focus on higher-level issues like resource
allocation and partition selection after addressing validation errors.

THESE ARE NOT ERRORS (do not mention them):
- Using --nodelist with a single node is VALID
- Using --ntasks=1 is VALID (even if it's the default)
- Using --nodes=1 is VALID
- Using --partition with --nodelist together is VALID
- --mem=32G is VALID (G suffix is correct)
- --time=04:00:00 is VALID (HH:MM:SS format)
- --gres=gpu:1 is VALID (colon syntax)

If you find NO validation errors and no higher-level issues, respond:
"The script syntax is correct. No errors found."

Only mention actual typos or invalid syntax. Do not suggest removing valid directives."""


# ─── Intent Detection & Prompt Building ───────────────────────────────────────

def detect_prompt_intent(messages: List[Dict[str, str]]) -> str:
    """
    Detect the high-level intent of the conversation to select prompt sections.
    Returns one of: "node_query", "script_write", "script_review", "script_adapt",
                    "job_query", "general"
    """
    adapt_keywords = [
        "adapt", "convert", "change", "different node", "alternative",
        "instead", "modify", "adjust", "migrate", "other node", "suggest",
        "replace", "available",
    ]
    write_keywords = [
        "write", "generate", "create", "make", "give me a script",
        "slurm script", "batch script", "sbatch script", "need a script",
        "new script",
    ]
    node_keywords = [
        "what nodes", "available nodes", "node status", "which nodes",
        "list nodes", "show nodes", "check nodes", "cluster nodes",
        "what resources", "available resources", "what gpu", "what cpu",
        "gpu nodes", "cpu nodes", "nodes are", "nodes available",
    ]
    job_keywords = [
        "job status", "my jobs", "job queue", "running jobs",
        "pending jobs", "submitted job", "jobs_list", "job_info",
    ]

    user_messages = [m for m in reversed(messages) if m.get("role") == "user"]

    for msg in user_messages[:3]:
        content = msg.get("content", "")
        if content.startswith("[SYSTEM DATA"):
            continue
        c = content.lower()
        has_script = "#sbatch" in c

        if has_script and any(kw in c for kw in adapt_keywords):
            return "script_adapt"
        if has_script:
            return "script_review"
        if any(kw in c for kw in write_keywords):
            return "script_write"
        if any(kw in c for kw in node_keywords):
            return "node_query"
        if any(kw in c for kw in job_keywords):
            return "job_query"

    return "general"


def build_system_prompt(intent: str) -> str:
    """Assemble a focused system prompt from sections based on detected intent."""
    sections_map = {
        "node_query":    [PROMPT_BASE, PROMPT_FETCH, PROMPT_NODES],
        "script_write":  [PROMPT_BASE, PROMPT_FETCH, PROMPT_NODES, PROMPT_SCRIPT_WRITE],
        "script_review": [PROMPT_BASE, PROMPT_SCRIPT_REVIEW],
        "script_adapt":  [PROMPT_BASE, PROMPT_FETCH, PROMPT_NODES, PROMPT_SCRIPT_WRITE],
        "job_query":     [PROMPT_BASE, PROMPT_FETCH],
        "general":       [PROMPT_BASE],
    }
    sections = sections_map.get(intent, [PROMPT_BASE, PROMPT_FETCH])
    return "\n\n".join(sections)


# Full prompt assembled from all sections (used as fallback)
SYSTEM_PROMPT = "\n\n".join([
    PROMPT_BASE, PROMPT_FETCH, PROMPT_NODES, PROMPT_SCRIPT_WRITE, PROMPT_SCRIPT_REVIEW
])


# ─── Client ───────────────────────────────────────────────────────────────────

class OllamaClient:
    """Client for interacting with Ollama LLM."""

    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.OLLAMA_MODEL
        self.timeout = config.OLLAMA_TIMEOUT

    async def chat(self, messages: List[Dict[str, str]], system_prompt: str | None = None) -> str:
        """Send messages to Ollama and get a response."""
        used_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        full_messages = [{"role": "system", "content": used_prompt}] + messages

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": False,
            "options": {"temperature": 0}
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]


ollama_client = OllamaClient()
