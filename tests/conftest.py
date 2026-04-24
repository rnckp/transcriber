from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import load_config
from app.main import app
from tests.stubs import StubService


@pytest.fixture
def stub_service() -> StubService:
    return StubService()


@pytest.fixture
def client(stub_service: StubService) -> TestClient:
    app.state.config = load_config(Path("config.yaml"))
    app.state.service = stub_service
    with TestClient(app) as test_client:
        yield test_client
