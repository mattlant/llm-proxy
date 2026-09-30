import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "provider_compliance: reusable provider extension contract checks")
