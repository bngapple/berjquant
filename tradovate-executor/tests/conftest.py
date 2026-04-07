"""
pytest configuration for tradovate-executor tests.
"""
import pytest


# Use auto mode so async test functions are detected without @pytest.mark.asyncio
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark a test as asyncio"
    )
