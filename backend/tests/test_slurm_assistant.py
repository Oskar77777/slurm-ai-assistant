"""
Tests for SLURM script generation and cluster queries.

Run with:
  pytest tests/test_slurm_assistant.py -v              # Offline (mock data)
  pytest tests/test_slurm_assistant.py -v --online    # Live API
"""

import pytest
import asyncio
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

    def test_gpu_script(self, async_client, services_available):
        """Generate a script for GPU training."""
        question = "I need a SLURM script for training a PyTorch model on a GPU. Can you help?"

        data = asyncio.run(ask_assistant(async_client, question))
        content = data["response"].lower()

        # Should contain SLURM directives
        assert any(ind in content for ind in ["#sbatch", "sbatch", "slurm"]), \
            "Expected SLURM directives"

        # Should contain GPU allocation
        assert any(ind in content for ind in ["--gres=gpu", "--gpus", "gpu"]), \
            "Expected GPU allocation"

    def test_cpu_script(self, async_client, services_available):
        """Generate a script for CPU-only job."""
        question = "Write me a SLURM script for a CPU-only job that needs 8 cores and 16GB RAM for 2 hours"

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
# Script Adaptation Tests
# =============================================================================

class TestScriptAdaptation:
    """Tests for suggesting alternative nodes and adapting scripts."""

    @pytest.mark.parametrize("script_name", [
        "gpu_training",
        "gpu_inference",
        "multi_gpu",
        "cpu_compute",
        "cpu_memory",
    ])
    def test_suggest_alternatives(self, async_client, services_available, script_name):
        """Ask for alternative nodes for a script."""
        script_data = load_script(script_name)
        node = script_data["node"]
        script_type = script_data["type"]
        script_content = script_data["script"].format(node=node)

        node_type = "GPU" if script_type == "gpu" else "compute"
        question = f"""I have this SLURM script that's hardcoded to run on node {node}, but that node might not be available. Can you suggest alternative {node_type} nodes?

{script_content}"""

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
        ("cpu_compute", "gpu"),
        ("cpu_memory", "gpu"),
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

        question = f"""I have this SLURM script written for {current_label} nodes, but only {target_label} nodes are available. Can you adapt it?

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
