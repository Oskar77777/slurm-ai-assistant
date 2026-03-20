import httpx
import logging
from typing import Optional, Any
import config

logger = logging.getLogger(__name__)


class Ex3Client:
    """Client for interacting with the eX3 cluster API."""

    def __init__(self):
        self.base_url = config.EX3_API_BASE_URL
        self.timeout = config.EX3_API_TIMEOUT
        self.default_cluster = config.EX3_DEFAULT_CLUSTER
        logger.info(f"Ex3Client initialized with base_url={self.base_url}, cluster={self.default_cluster}")

    async def _request(self, endpoint: str) -> dict[str, Any]:
        """Make a GET request to the eX3 API."""
        url = f"{self.base_url}/{endpoint}"
        logger.info(f"Making request to: {url}")
        try:
            async with httpx.AsyncClient(verify=False, timeout=self.timeout) as client:
                response = await client.get(url)
                logger.info(f"Response status: {response.status_code}")
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError as e:
            logger.error(f"Connection error to {url}: {e}")
            raise Exception(f"Cannot connect to eX3 API at {url}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e}")
            raise Exception(f"eX3 API returned error {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            raise

    async def get_clusters(self) -> dict[str, Any]:
        """Get list of available clusters."""
        return await self._request("cluster")

    async def get_nodes(self, cluster: Optional[str] = None) -> dict[str, Any]:
        """Get all nodes in a cluster."""
        cluster = cluster or self.default_cluster
        return await self._request(f"cluster/{cluster}/nodes")

    async def get_node_info(self, nodename: str, cluster: Optional[str] = None) -> dict[str, Any]:
        """Get specific node information."""
        cluster = cluster or self.default_cluster
        return await self._request(f"cluster/{cluster}/nodes/{nodename}/info")

    async def get_all_nodes_info(self, cluster: Optional[str] = None) -> dict[str, Any]:
        """Get detailed system info for all nodes."""
        cluster = cluster or self.default_cluster
        return await self._request(f"cluster/{cluster}/nodes/info")

    async def get_jobs(self, cluster: Optional[str] = None) -> dict[str, Any]:
        """Get all jobs in the cluster."""
        cluster = cluster or self.default_cluster
        return await self._request(f"cluster/{cluster}/jobs")

    async def get_job_info(self, job_id: str, cluster: Optional[str] = None) -> dict[str, Any]:
        """Get specific job information."""
        cluster = cluster or self.default_cluster
        return await self._request(f"cluster/{cluster}/jobs/{job_id}?epoch=0")

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


ex3_client = Ex3Client()
