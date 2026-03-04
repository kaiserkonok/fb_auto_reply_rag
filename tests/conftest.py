"""
Pytest configuration and fixtures.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Create test client."""
    from wsgi import create_app
    app = create_app()
    return app.test_client()
