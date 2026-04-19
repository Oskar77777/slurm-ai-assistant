import httpx
from typing import List, Dict, Any
import config


# ─── Prompt Sections ──────────────────────────────────────────────────────────

PROMPT_BASE = """You are an AI assistant helping users with SLURM scripts and eX3 cluster status at Simula.

You help users:
- Write SLURM batch scripts
- Understand cluster resources and availability
"""


PROMPT_FETCH = """FETCHING CLUSTER DATA:
In most cases cluster data is already provided in the conversation as [SYSTEM DATA]. Use it directly.
If you need data that is NOT already in the conversation, respond with ONLY a fetch command:

Available fetch commands:
- [FETCH: cluster_list] - Get list of available clusters
- [FETCH: nodes_list] - Get all nodes in eX3 cluster (summarized with availability)
- [FETCH: node_info:NODENAME] - Get specific node details (replace NODENAME with actual node name)
- [FETCH: jobs_list] - Get all jobs in the cluster (running, pending, completed)
- [FETCH: job_info:JOBID] - Get specific job details (replace JOBID with actual job ID)
- [FETCH: partitions_list] - Get all partitions with node membership, CPU/GPU counts, and job queue sizes

RULES:
1. To fetch data, respond with ONLY the fetch command (nothing else)
2. When you see "[SYSTEM DATA" in a user message, that contains REAL cluster data - use it!
3. NEVER generate or fabricate [SYSTEM DATA] blocks yourself. You cannot know real cluster state.
4. NEVER invent node names, partition names, GPU counts, or availability.
"""


PROMPT_NODES = """NODE DATA FORMAT:
Node data is pre-formatted as human-readable text showing:
- Summary with counts (total, idle, partial, full)
- IDLE: Fully available nodes with resources
- PARTIAL: Nodes with some resources in use
- FULLY OCCUPIED: Nodes with no available resources

Each node line shows: name, available/total CPUs, GPUs (if applicable), RAM, and partitions.
The values in square brackets at the end of each node line are the partition names — use one of these directly as the --partition value in any SLURM script you write.

NODE NAMING:
- Nodes starting with 'n' are CPU compute nodes, 'g' are GPU nodes, 'gh' are high-memory GPU nodes

REQUIRED OUTPUT FORMAT FOR NODE LISTINGS:
Use this table format ONLY when listing multiple nodes. For a single node detail request, present the information as a plain readable summary instead.

When listing multiple nodes, you MUST use exactly this structure:

### 🟢 Idle (N nodes)
| Node | CPUs | GPUs | RAM | Partitions |
|------|------|------|-----|------------|
| name | avail/total | N x GPU_MODEL | X GB | partition1, partition2 |

### 🟡 Partially Occupied (N nodes)
| Node | CPUs | GPUs | RAM | Partitions |
|------|------|------|-----|------------|
| name | avail/total | N x GPU_MODEL | X GB | partition1 |

### 🔴 Fully Occupied (N nodes)
| Node | Partitions |
|------|------------|
| name | partition1 |

Rules:
- ALWAYS include all three sections, even if a section has 0 nodes (write "none" as the only row)
- For CPU-only nodes (no GPU in the data), write — in the GPU column. NEVER put the RAM value there.
- Every row MUST have exactly 5 cells: Node | CPUs | GPUs | RAM | Partitions
- List every node — never abbreviate with "..." or skip entries
"""


PROMPT_PARTITIONS = """PARTITION DATA FORMAT:
When listing partitions, you MUST present them as a markdown table using exactly this format:

| Partition | Nodes | CPUs | GPUs | Jobs Running | Jobs Pending |
|-----------|-------|------|------|--------------|--------------|
| name | N [compact-range] | N | N (X reserved, Y in use) | N | N |

Rules:
- For partitions with no GPUs, write — in the GPUs column
- List every partition — never skip or abbreviate
- Use the compact node range (e.g. n[001-004]) in the Nodes column
- Do not add any extra text between partition rows
"""


PROMPT_SCRIPT_WRITE = """WRITING SLURM SCRIPTS:
Always write a proper batch script with #SBATCH directives at the top — never call `sbatch` from inside a script.

Every SLURM script MUST include the following directives:
```
#SBATCH --job-name=<name>
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --time=<HH:MM:SS>
#SBATCH --partition=<partition>
#SBATCH --nodes=<N>
#SBATCH --ntasks=<N>
#SBATCH --mem=<N>G
#SBATCH --cpus-per-task=<N>
#SBATCH --gpus-per-node=<N>
```
For GPU jobs, you MUST add --gpus-per-node=N using the value from the FEASIBLE OPTIONS block.
Set --mem to a reasonable estimate for the job (e.g. 32G–128G), never to the full node memory.

When the cluster data contains a [RESOURCE RECOMMENDATION FOR N GPUs] block, use ONLY the values
listed under FEASIBLE OPTIONS for --partition, --nodes, and --gpus-per-node. Do not recalculate these yourself.

Multi-node distributed training:
- Use --nodes=N where N is the number of nodes needed
- Use --gpus-per-node to specify GPUs per node (not total GPUs)
- Use torchrun with --nnodes=$SLURM_NNODES and --rdzv-backend=c10d for PyTorch distributed jobs

If a user asks for X total GPUs and the available nodes have Y GPUs each, set --nodes=ceil(X/Y) and --gpus-per-node=Y.
"""


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


PROMPT_REMINDER = """You are an AI assistant for the eX3 HPC cluster at Simula. Continue the conversation.
- Cluster data has already been provided earlier in this conversation — use it.
- NEVER fabricate node names, partition names, GPU counts, or [SYSTEM DATA] blocks.
- If you genuinely need fresh data not already in context, use [FETCH: nodes_list] or similar."""


# ─── Intent Detection & Prompt Building ───────────────────────────────────────

_ADAPT_KEYWORDS = [
    "adapt", "convert", "change", "different node", "alternative",
    "instead", "modify", "adjust", "migrate", "other node", "suggest",
    "replace", "available",
]
_WRITE_KEYWORDS = [
    "write", "generate", "create", "make", "give me a script",
    "slurm script", "batch script", "sbatch script", "need a script",
    "new script",
]
_NODE_KEYWORDS = [
    "what nodes", "available nodes", "node status", "which nodes",
    "list nodes", "show nodes", "check nodes", "cluster nodes",
    "what resources", "available resources", "what gpu", "what cpu",
    "gpu nodes", "cpu nodes", "nodes are", "nodes available",
    "list all nodes", "show all nodes", "all nodes", "current nodes",
    "nodes right now", "nodes currently", "cluster status", "cluster resources",
    "free nodes", "idle nodes", "show cluster", "what is available",
    "what's available", "what are the nodes", "show me nodes",
]
_JOB_KEYWORDS = [
    "job status", "my jobs", "job queue", "running jobs",
    "pending jobs", "submitted job", "jobs_list", "job_info",
    "list jobs", "show jobs", "check jobs", "what jobs",
]
_PARTITION_KEYWORDS = [
    "partition", "partitions", "list partitions", "show partitions",
    "available partitions", "what partitions", "which partitions",
    "partition names", "partition list",
]


def _classify_message(content: str) -> str:
    """Classify a single message string into an intent."""
    c = content.lower()
    has_script = "#sbatch" in c
    if has_script and any(kw in c for kw in _ADAPT_KEYWORDS):
        return "script_adapt"
    if has_script:
        return "script_review"
    if any(kw in c for kw in _WRITE_KEYWORDS):
        return "script_write"
    if any(kw in c for kw in _NODE_KEYWORDS):
        return "node_query"
    if any(kw in c for kw in _JOB_KEYWORDS):
        return "job_query"
    if any(kw in c for kw in _PARTITION_KEYWORDS):
        return "partition_query"
    return "general"


def detect_fetch_intent(messages: List[Dict[str, str]]) -> str:
    """
    Detect intent from the LATEST user message only.
    Used to decide whether to pre-fetch cluster data — must not look back at
    previous messages or a follow-up like 'nice, thank you!' would re-trigger a fetch.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if content.startswith("[SYSTEM DATA"):
            continue
        return _classify_message(content)
    return "general"


def detect_prompt_intent(messages: List[Dict[str, str]]) -> str:
    """
    Detect intent by scanning the last 3 user messages.
    Used for system prompt selection — the lookback helps pick the right prompt
    sections for follow-up messages that reference earlier context.
    Do NOT use this for pre-fetch decisions (use detect_fetch_intent instead).
    """
    user_messages = [m for m in reversed(messages) if m.get("role") == "user"]
    for msg in user_messages[:3]:
        content = msg.get("content", "")
        if content.startswith("[SYSTEM DATA"):
            continue
        intent = _classify_message(content)
        if intent != "general":
            return intent
    return "general"


def build_system_prompt(intent: str, first_turn: bool = True) -> str:
    """Assemble a system prompt based on intent and whether this is the first turn."""
    if not first_turn:
        # Subsequent turns: short reminder + only the action-relevant section
        action_sections = {
            "node_query":      [PROMPT_REMINDER, PROMPT_NODES],
            "script_write":    [PROMPT_REMINDER, PROMPT_SCRIPT_WRITE],
            "script_adapt":    [PROMPT_REMINDER, PROMPT_NODES, PROMPT_SCRIPT_WRITE],
            "script_review":   [PROMPT_REMINDER, PROMPT_SCRIPT_REVIEW],
            "partition_query": [PROMPT_REMINDER, PROMPT_PARTITIONS],
        }
        return "\n\n".join(action_sections.get(intent, [PROMPT_REMINDER]))

    # First turn: full intent-specific prompt
    sections_map = {
        "node_query":      [PROMPT_BASE, PROMPT_FETCH, PROMPT_NODES],
        "script_write":    [PROMPT_BASE, PROMPT_FETCH, PROMPT_NODES, PROMPT_SCRIPT_WRITE],
        "script_review":   [PROMPT_BASE, PROMPT_SCRIPT_REVIEW],
        "script_adapt":    [PROMPT_BASE, PROMPT_FETCH, PROMPT_NODES, PROMPT_SCRIPT_WRITE],
        "job_query":       [PROMPT_BASE, PROMPT_FETCH],
        "partition_query": [PROMPT_BASE, PROMPT_FETCH, PROMPT_PARTITIONS],
        "general":         [PROMPT_BASE, PROMPT_FETCH],
    }
    return "\n\n".join(sections_map.get(intent, [PROMPT_BASE, PROMPT_FETCH]))


# Full prompt assembled from all sections (used as fallback)
SYSTEM_PROMPT = "\n\n".join([
    PROMPT_BASE, PROMPT_FETCH, PROMPT_NODES, PROMPT_PARTITIONS, PROMPT_SCRIPT_WRITE, PROMPT_SCRIPT_REVIEW
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
            "stream": False
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]


ollama_client = OllamaClient()
