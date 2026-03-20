"""
Shared pytest fixtures for integration tests.

Usage:
  pytest tests/ -v                    # Mock data (default/offline)
  pytest tests/ --offline -v          # Mock data (explicit)
  pytest tests/ --online -v           # Live eX3 API

Prerequisites:
- Ollama must be running (ollama serve)
- For --online: SSH tunnel to eX3 must be active
"""

import pytest
import httpx
from httpx import ASGITransport
from unittest.mock import patch

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

import config
from tests.mocks.mock_ex3_client import mock_ex3_client
from tests.mocks.logging_ex3_client import LoggingEx3Client


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--online",
        action="store_true",
        default=False,
        help="Use live eX3 API"
    )
    parser.addoption(
        "--offline",
        action="store_true",
        default=False,
        help="Use mock eX3 data (default)"
    )


@pytest.fixture(scope="session")
def use_live_api(request):
    """Check if tests should use live eX3 API."""
    return request.config.getoption("--online")


@pytest.fixture
def context():
    """Shared context for BDD steps to pass data between steps."""
    class Context:
        response = None
        messages = []
    return Context()


@pytest.fixture(autouse=True)
def mock_ex3_api(use_live_api):
    """Mock the eX3 client unless --online flag is used."""
    if use_live_api:
        # Online mode: wrap real client with logging
        from services.ex3_client import ex3_client
        logging_client = LoggingEx3Client(ex3_client)
        with patch("main.ex3_client", logging_client):
            yield
    else:
        # Offline mode: use mock data
        with patch("main.ex3_client", mock_ex3_client):
            yield


@pytest.fixture
def async_client():
    """Factory fixture that creates fresh AsyncClient for each async request."""
    from main import app

    class ClientFactory:
        def __init__(self):
            self.transport = ASGITransport(app=app)

        async def post(self, *args, **kwargs):
            async with httpx.AsyncClient(transport=self.transport, base_url="http://test") as client:
                return await client.post(*args, **kwargs)

        async def get(self, *args, **kwargs):
            async with httpx.AsyncClient(transport=self.transport, base_url="http://test") as client:
                return await client.get(*args, **kwargs)

    return ClientFactory()


@pytest.fixture(scope="session")
def check_ollama():
    """Verify Ollama is accessible before running tests."""
    try:
        response = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if response.status_code != 200:
            pytest.skip(f"Ollama returned status {response.status_code}")
    except httpx.ConnectError:
        pytest.skip("Ollama is not running. Start with: ollama serve")
    except Exception as e:
        pytest.skip(f"Cannot connect to Ollama: {e}")


@pytest.fixture(scope="session")
def check_ex3(use_live_api):
    """Placeholder for BDD background step."""
    mode = "ONLINE (live API)" if use_live_api else "OFFLINE (mock data)"
    print(f"\n[Setup] Mode: {mode}")


@pytest.fixture
def services_available(check_ollama, check_ex3):
    """Composite fixture ensuring all required services are available."""
    return True
