import pytest

from app.services import capture_cache, gemini_client, request_timing, visitor_store


@pytest.fixture(autouse=True)
def _no_real_visitor_records(monkeypatch):
    """Tests that call /api/tryon must not add rows to the real data/visitors.db."""
    monkeypatch.setattr(visitor_store, "VISITOR_DATA_ENABLED", False)


@pytest.fixture(autouse=True)
def _no_capture_carryover():
    """Tests reuse the same small images with different mocks: a capture cached by one must not reach the next."""
    capture_cache.clear()
    gemini_client.clear_user_appearance()


@pytest.fixture(autouse=True)
def _no_timing_log(monkeypatch):
    """Tests that call /api/tryon must not append rows to the real logs/tryon_timings.csv."""
    monkeypatch.setattr(request_timing, "TRYON_TIMING_LOG", False)
