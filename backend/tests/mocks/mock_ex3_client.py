"""
Mock eX3 client that loads data from JSON files instead of calling the real API.
"""

import json
from pathlib import Path
from typing import Any

# Path to mock data files
MOCK_DATA_DIR = Path(__file__).parent.parent.parent / "api_mock_data"

# Simulated base URL (matches real API structure)
MOCK_BASE_URL = "https://localhost:12200/api/v2"
MOCK_CLUSTER = "ex3"


class MockEx3Client:
    """Mock client that returns data from JSON files."""

    def __init__(self):
        self.mock_data_dir = MOCK_DATA_DIR
        self._cache = {}
        self.base_url = MOCK_BASE_URL
        self.default_cluster = MOCK_CLUSTER

    def _log_endpoint(self, endpoint: str):
        """Print the full URL that would be called."""
        url = f"{self.base_url}/{endpoint}"
        print(f"\n{'=' * 80}")
        print(f"API ENDPOINT: {url}")
        print(f"(Using mock data)")
        print(f"{'=' * 80}\n")

    def _load_json(self, filename: str) -> Any:
        """Load and cache JSON data from a file."""
        if filename not in self._cache:
            file_path = self.mock_data_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Mock data file not found: {file_path}")
            with open(file_path, 'r') as f:
                self._cache[filename] = json.load(f)
        return self._cache[filename]

    async def get_clusters(self) -> list[dict[str, Any]]:
        """Get list of available clusters (mocked)."""
        self._log_endpoint("cluster")
        nodes = self._load_json("nodes.json")
        return [{
            "time": "2026-02-23T11:00:00Z",
            "cluster": "ex3.simula.no",
            "slurm": 1,
            "nodes": nodes
        }]

    async def get_nodes(self, cluster: str = None) -> list[str]:
        """Get all node names in a cluster (mocked)."""
        cluster = cluster or self.default_cluster
        self._log_endpoint(f"cluster/{cluster}/nodes")
        return self._load_json("nodes.json")

    async def get_node_info(self, nodename: str, cluster: str = None) -> dict[str, Any]:
        """Get specific node information (mocked)."""
        cluster = cluster or self.default_cluster
        self._log_endpoint(f"cluster/{cluster}/nodes/{nodename}/info")
        # First try nodeById.json
        try:
            node_data = self._load_json("nodeById.json")
            if nodename in node_data:
                return {nodename: node_data[nodename]}
        except FileNotFoundError:
            pass

        # Fall back to nodesInfo.json
        all_nodes = self._load_json("nodesInfo.json")
        if nodename in all_nodes:
            return {nodename: all_nodes[nodename]}

        raise ValueError(f"Node {nodename} not found in mock data")

    async def get_all_nodes_info(self, cluster: str = None) -> dict[str, Any]:
        """Get detailed system info for all nodes (mocked)."""
        cluster = cluster or self.default_cluster
        self._log_endpoint(f"cluster/{cluster}/nodes/info")
        return self._load_json("nodesInfo.json")

    async def get_partitions(self, cluster: str = None) -> list[dict[str, Any]]:
        """Get partition information (mocked)."""
        cluster = cluster or self.default_cluster
        self._log_endpoint(f"cluster/{cluster}/partitions")
        return self._load_json("partitions.json")

    async def get_jobs(self, cluster: str = None) -> dict[str, Any]:
        """Get jobs information (mocked)."""
        cluster = cluster or self.default_cluster
        self._log_endpoint(f"cluster/{cluster}/jobs")
        return self._load_json("jobs.json")

    async def get_job_info(self, job_id: str, cluster: str = None) -> dict[str, Any]:
        """Get specific job information (mocked)."""
        cluster = cluster or self.default_cluster
        self._log_endpoint(f"cluster/{cluster}/jobs/{job_id}?epoch=0")
        return self._load_json("jobById.json")

    async def call_by_tool_name(self, tool_name: str) -> dict[str, Any]:
        """Call an API endpoint based on the tool name from LLM (mocked)."""
        if tool_name == "cluster_list":
            return await self.get_clusters()
        elif tool_name == "nodes_list":
            return await self.get_all_nodes_info()
        elif tool_name == "nodes_info":
            return await self.get_all_nodes_info()
        elif tool_name.startswith("node_info:"):
            nodename = tool_name.split(":", 1)[1]
            return await self.get_node_info(nodename)
        elif tool_name == "jobs_list":
            return await self.get_jobs()
        elif tool_name.startswith("job_info:"):
            job_id = tool_name.split(":", 1)[1]
            return await self.get_job_info(job_id)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


# Singleton instance for easy importing
mock_ex3_client = MockEx3Client()
