import time

import pytest

from app.services.rate_limiter import InMemoryRateLimiter


class TestInMemoryRateLimiter:
    def test_allows_requests_within_limit(self):
        limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)
        assert limiter.is_allowed("192.168.1.1")
        assert limiter.is_allowed("192.168.1.1")
        assert limiter.is_allowed("192.168.1.1")

    def test_blocks_when_limit_exceeded(self):
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("10.0.0.1")
        assert limiter.is_allowed("10.0.0.1")
        assert not limiter.is_allowed("10.0.0.1")

    def test_different_ips_have_separate_limits(self):
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
        assert limiter.is_allowed("1.1.1.1")
        assert limiter.is_allowed("2.2.2.2")

    def test_window_expires(self):
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=1)
        assert limiter.is_allowed("3.3.3.3")
        assert not limiter.is_allowed("3.3.3.3")
        time.sleep(1.1)
        assert limiter.is_allowed("3.3.3.3")

    def test_cleanup_removes_expired_entries(self):
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=0)
        limiter.is_allowed("4.4.4.4")
        limiter.cleanup()
        assert len(limiter._requests) == 0

    def test_get_remaining_returns_correct_count(self):
        limiter = InMemoryRateLimiter(max_requests=5, window_seconds=60)
        assert limiter.get_remaining("5.5.5.5") == 5
        limiter.is_allowed("5.5.5.5")
        assert limiter.get_remaining("5.5.5.5") == 4
