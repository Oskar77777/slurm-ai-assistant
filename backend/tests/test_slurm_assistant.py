"""
Tests for SLURM script generation and cluster queries.

Run with:
  pytest tests/test_slurm_assistant.py -v              # Offline (mock data)
  pytest tests/test_slurm_assistant.py -v --online    # Live API
"""

import pytest
import asyncio
import re
import yaml
from pathlib import Path


# Path to script fixtures
SCRIPTS_DIR = Path(__file__).parent / "fixtures" / "scripts"


def print_response(question: str, response: str):
    """Print the question and LLM response for visibility."""
    print("\n" + "=" * 80)
    print("QUESTION:")
    print("-" * 80)
    print(question)
    print("\n" + "-" * 80)
    print("LLM RESPONSE:")
    print("-" * 80)
    print(response)
    print("=" * 80 + "\n")


async def ask_assistant(async_client, question: str) -> dict:
    """Send a question to the assistant and return the response."""
    response = await async_client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": question}]},
        timeout=120.0
    )
    assert response.status_code == 200, f"API returned {response.status_code}"
    data = response.json()
    assert "response" in data, "Missing 'response' field"
    assert len(data["response"]) > 0, "Empty response"

    print_response(question, data["response"])
    return data


def load_script(script_name: str) -> dict:
    """Load a script fixture. Returns dict with: node, type, script"""
    script_path = SCRIPTS_DIR / f"{script_name}.yaml"
    if not script_path.exists():
        available = [f.stem for f in SCRIPTS_DIR.glob("*.yaml")]
        raise ValueError(f"Script '{script_name}' not found. Available: {available}")

    with open(script_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Cluster Query Tests
# =============================================================================

class TestClusterQueries:
    """Tests for querying cluster information."""

    def test_list_nodes(self, async_client, services_available):
        """Ask what nodes are available."""
        question = "What nodes are available in the cluster right now?"

        data = asyncio.run(ask_assistant(async_client, question))
        content = data["response"].lower()

        # Should mention nodes
        node_indicators = ["node", "n0", "g0", "gh0", "compute", "gpu"]
        assert any(ind in content for ind in node_indicators), \
            "Expected mention of cluster nodes"

    def test_gpu_availability(self, async_client, services_available):
        """Ask about GPU nodes and specifications."""
        question = "What GPU nodes are currently available on the eX3 cluster and what are their specifications?"

        data = asyncio.run(ask_assistant(async_client, question))
        content = data["response"].lower()

        # Should reference real cluster data
        cluster_indicators = ["gpu", "node", "available", "a100", "v100", "h100", "memory", "cpus"]
        assert any(ind in content for ind in cluster_indicators), \
            "Expected cluster-related content"


# =============================================================================
# Script Generation Tests
# =============================================================================

class TestScriptGeneration:
    """Tests for generating SLURM scripts."""

    # Known eX3 partition names from the cluster
    EX3_GPU_PARTITIONS = ["dgx2q", "hgx2q", "h200q", "gh200q", "a100q", "a40q",
                          "mi210q", "mi100q", "mi50q", "milanq", "huaq", "defq", "xeonmaxq"]
    EX3_CPU_PARTITIONS = ["armq", "fpgaq", "genoaxq", "rome16q", "slowq",
                          "virtq", "flowq", "aarchq", "ipuq"]

    def test_gpu_script(self, async_client, services_available):
        """Generate a GPU script based on currently available eX3 nodes."""
        question = (
            "I need a SLURM script for training a PyTorch model on a GPU on the eX3 cluster. "
            "Can you check what GPU nodes are currently available and generate an appropriate script?"
        )

        data = asyncio.run(ask_assistant(async_client, question))
        content = data["response"].lower()

        # Should contain SLURM directives
        assert any(ind in content for ind in ["#sbatch", "sbatch", "slurm"]), \
            "Expected SLURM directives"

        # Should contain GPU allocation
        assert any(ind in content for ind in ["--gres=gpu", "--gpus", "gpu"]), \
            "Expected GPU allocation"

        # Should reference a real eX3 GPU partition (proving it used cluster data)
        assert any(p in content for p in self.EX3_GPU_PARTITIONS), \
            f"Expected script to reference a real eX3 GPU partition. Known GPU partitions: {self.EX3_GPU_PARTITIONS}"

    def test_cpu_script(self, async_client, services_available):
        """Generate a CPU script based on currently available eX3 nodes."""
        question = (
            "I need a SLURM script for a CPU-only job that needs 8 cores and 16GB RAM "
            "for 2 hours on the eX3 cluster. Can you check what CPU nodes are currently "
            "available and generate an appropriate script?"
        )

        data = asyncio.run(ask_assistant(async_client, question))
        content = data["response"].lower()

        # Should contain SLURM directives
        assert any(ind in content for ind in ["#sbatch", "sbatch", "slurm"]), \
            "Expected SLURM directives"

        # Should contain time allocation
        assert any(ind in content for ind in ["--time", "time=", "hour"]), \
            "Expected time allocation"

        # Should contain memory allocation
        assert any(ind in content for ind in ["--mem", "memory", "gb", "ram"]), \
            "Expected memory allocation"

        # Should reference a real eX3 CPU partition (proving it used cluster data)
        assert any(p in content for p in self.EX3_CPU_PARTITIONS), \
            f"Expected script to reference a real eX3 CPU partition. Known CPU partitions: {self.EX3_CPU_PARTITIONS}"


# =============================================================================
# Script Review Tests
# =============================================================================

# Scripts to review with their expected typos/errors (empty list = no errors)
REVIEW_TEST_CASES = {
    "correct_script": {
        "script": """#!/bin/bash
#SBATCH --job-name=pytorch_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

python train.py""",
        "expected_errors": []
    },
    "typo_cpus_and_mem": {
        "script": """#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-taski=4
#SBATCH --mem=8GG
#SBATCH --time=01:00:00

python my_script.py""",
        "expected_errors": ["cpus-per-taski", "8gg"]
    },
    "typo_nodes_and_time": {
        "script": """#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output_%j.txt
#SBATCH --nodess=1
#SBATCH --ntasks=11111
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GF
#SBATCH --timerr=01:00:00

python my_script.py""",
        "expected_errors": ["nodess", "timer"]
    },
}


class TestScriptReview:
    """Tests for reviewing SLURM scripts for typos and syntax errors."""

    @pytest.mark.parametrize("test_case", REVIEW_TEST_CASES.keys())
    def test_script_review(self, async_client, services_available, test_case):
        """User submits a script and asks if it looks good."""
        test_data = REVIEW_TEST_CASES[test_case]
        script = test_data["script"]
        expected_errors = test_data["expected_errors"]

        question = f"""I wrote this SLURM batch script. Does everything look good?

{script}"""

        data = asyncio.run(ask_assistant(async_client, question))
        content = data["response"].lower()

        # Check that each expected error is mentioned in the response
        for error in expected_errors:
            assert error.lower() in content, \
                f"Expected LLM to identify '{error}' but it was not mentioned"


# =============================================================================
# Multi-Node Distributed Training Tests
# =============================================================================

class TestMultiNodeTraining:
    """Tests for generating multi-node distributed training SLURM scripts."""

    EX3_GPU_PARTITIONS = ["dgx2q", "hgx2q", "h200q", "gh200q", "a100q", "a40q",
                          "mi210q", "mi100q", "mi50q", "xeonmaxq"]

    def test_multinode_gpu_script(self, async_client, services_available):
        """Generate a multi-node distributed training script for 12 GPUs."""
        question = (
            "I need to fine-tune a large PyTorch model that requires 8 GPUs. "
            "Can you check what GPU nodes are currently available on the eX3 cluster "
            "and write a SLURM script for distributed multi-node training?"
        )

        data = asyncio.run(ask_assistant(async_client, question))
        content = data["response"].lower()
        raw = data["response"]

        # Should contain a SLURM script
        assert any(ind in content for ind in ["#!/bin/bash", "#sbatch"]), \
            "Expected a SLURM script"

        # Should request more than one node
        nodes_match = re.search(r"--nodes[= ](\d+)", raw)
        assert nodes_match is not None, \
            "Expected --nodes directive in script"
        num_nodes = int(nodes_match.group(1))
        assert num_nodes > 1, \
            f"Expected --nodes > 1 for multi-node job, got --nodes={num_nodes}"

        # Should allocate GPUs
        assert any(ind in content for ind in ["--gpus-per-node", "--gres=gpu", "--gpus"]), \
            "Expected GPU allocation directive"

        # Should use torchrun for PyTorch distributed training
        assert "torchrun" in content, \
            "Expected torchrun for PyTorch distributed training"

        # torchrun should be configured for multi-node (not --standalone)
        multi_node_indicators = ["nnodes", "slurm_nnodes", "rdzv", "nproc_per_node"]
        assert any(ind in content for ind in multi_node_indicators), \
            "Expected multi-node torchrun configuration (--nnodes, --rdzv-backend, etc.)"

        # Should reference a real eX3 GPU partition inside the script block itself
        script_match = re.search(r"```bash\n(.*?)```", raw, re.DOTALL)
        assert script_match is not None, "Expected a bash code block in the response"
        script_block = script_match.group(1).lower()
        assert any(p in script_block for p in self.EX3_GPU_PARTITIONS), \
            f"Expected a real eX3 GPU partition inside the script block. Known: {self.EX3_GPU_PARTITIONS}"


# =============================================================================
# Script Adaptation Tests
# =============================================================================

class TestScriptAdaptation:
    """Tests for suggesting alternative nodes and adapting scripts."""

    def test_suggest_alternatives(self, async_client, services_available):
        """Ask for alternative GPU nodes when the hardcoded node is fully occupied."""
        node = "g001"
        script = """#!/bin/bash
#SBATCH --job-name=gpu_training
#SBATCH --output=output_%j.txt
#SBATCH --nodelist=g001
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G

module load cuda/12.0
python train.py"""

        question = f"""I have this SLURM script hardcoded to run on node {node}, but it's fully occupied right now. Can you check what other GPU nodes are available on the eX3 cluster and suggest an alternative script I can run?

{script}"""

        data = asyncio.run(ask_assistant(async_client, question))
        content = data["response"].lower()

        # Should acknowledge the node
        assert any(ind in content for ind in [node.lower(), "nodelist", "hardcoded", "specific"]), \
            f"Expected to acknowledge node {node}"

        # Should suggest alternatives
        alt_indicators = ["alternative", "instead", "other", "available", "could use",
                         "can use", "recommend", "suggest", "partition"]
        assert any(ind in content for ind in alt_indicators), \
            "Expected to suggest alternatives"

    @pytest.mark.parametrize("script_name,target_type", [
        ("gpu_training", "cpu"),
        ("gpu_inference", "cpu"),
    ])
    def test_adapt_script(self, async_client, services_available, script_name, target_type):
        """Adapt a script for a different node type."""
        script_data = load_script(script_name)
        node = script_data["node"]
        current_type = script_data["type"]
        script_content = script_data["script"].format(node=node)

        current_label = "CPU" if current_type == "cpu" else "GPU"
        target_label = "GPU" if target_type == "gpu" else "CPU"

        question = f"""I have this SLURM script written for {current_label} nodes, but I need to run it on a {target_label} node instead. Can you check what {target_label} nodes are currently available on the cluster and adapt the script to use one of them?

{script_content}"""

        data = asyncio.run(ask_assistant(async_client, question))
        content = data["response"].lower()

        # Should provide a modified script
        assert any(ind in content for ind in ["#!/bin/bash", "#sbatch"]), \
            "Expected a modified SLURM script"

        # Should have appropriate directives
        if target_type == "gpu":
            assert any(ind in content for ind in ["--gres=gpu", "--gpus", "gpu", "cuda"]), \
                "Expected GPU directives"
        else:
            assert any(ind in content for ind in ["--cpus", "cpus-per-task", "cpu", "--ntasks"]), \
                "Expected CPU directives"

        # The adapted script should replace --nodelist with an appropriate node.
        # Find --nodelist lines in the adapted script (not comment lines).
        nodelist_lines = [
            line for line in content.split('\n')
            if '#sbatch' in line and 'nodelist' in line
        ]
        for line in nodelist_lines:
            assert node.lower() not in line, (
                f"LLM kept original {current_label} node '{node}' in --nodelist "
                f"when adapting to {target_label}. Expected it to fetch currently "
                f"available {target_label} nodes and replace the node."
            )
