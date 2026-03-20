"""
Logging wrapper for the real eX3 client.
Used in --online mode to show API calls and responses in test output.
"""

import json
from typing import Any


class LoggingEx3Client:
    """Wrapper that logs API calls and responses."""

    def __init__(self, real_client):
        self._client = real_client
        self.base_url = real_client.base_url
        self.default_cluster = real_client.default_cluster

    def _print_response(self, endpoint: str, data: Any):
        """Print the endpoint and first 50 lines of response."""
        url = f"{self.base_url}/{endpoint}"

        print("\n" + "=" * 80)
        print(f"API ENDPOINT: {url}")
        print("=" * 80)

        # Format JSON and limit to 50 lines
        if isinstance(data, (dict, list)):
            formatted = json.dumps(data, indent=2)
        else:
            formatted = str(data)

        lines = formatted.split('\n')
        preview = '\n'.join(lines[:500])

        print("RESPONSE (first 50 lines):")
        print("-" * 80)
        print(preview)

        if len(lines) > 50:
            print(f"\n... ({len(lines) - 50} more lines)")

        print("=" * 80 + "\n")

    async def get_clusters(self) -> dict[str, Any]:
        """Get list of available clusters."""
        data = await self._client.get_clusters()
        self._print_response("cluster", data)
        return data

    async def get_nodes(self, cluster: str = None) -> dict[str, Any]:
        """Get all nodes in a cluster."""
        cluster = cluster or self.default_cluster
        data = await self._client.get_nodes(cluster)
        self._print_response(f"cluster/{cluster}/nodes", data)
        return data

    async def get_node_info(self, nodename: str, cluster: str = None) -> dict[str, Any]:
        """Get specific node information."""
        cluster = cluster or self.default_cluster
        data = await self._client.get_node_info(nodename, cluster)
        self._print_response(f"cluster/{cluster}/nodes/{nodename}/info", data)
        return data

    async def get_all_nodes_info(self, cluster: str = None) -> dict[str, Any]:
        """Get detailed system info for all nodes."""
        cluster = cluster or self.default_cluster
        data = await self._client.get_all_nodes_info(cluster)
        self._print_response(f"cluster/{cluster}/nodes/info", data)
        return data

    async def get_jobs(self, cluster: str = None) -> dict[str, Any]:
        """Get all jobs in the cluster."""
        cluster = cluster or self.default_cluster
        data = await self._client.get_jobs(cluster)
        self._print_response(f"cluster/{cluster}/jobs", data)
        return data

    async def get_job_info(self, job_id: str, cluster: str = None) -> dict[str, Any]:
        """Get specific job information."""
        cluster = cluster or self.default_cluster
        data = await self._client.get_job_info(job_id, cluster)
        self._print_response(f"cluster/{cluster}/jobs/{job_id}?epoch=0", data)
        return data

    async def call_by_tool_name(self, tool_name: str) -> dict[str, Any]:
        """Call an API endpoint based on the tool name from LLM."""
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
