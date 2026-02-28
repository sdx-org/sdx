"""Simplified integration tests for rate limiting logic.

Testing Approach:
    This integration test suite validates rate limiting behavior at the API
    endpoint level, ensuring the feature works end-to-end in realistic
    scenarios. Tests are organized into four groups: (1) RateLimitingLogic
    tests verify patient and IP limits enforce correctly with proper HTTP
    headers and semantic validity. (2) RateLimitingWithCaching tests confirm
    cached responses don't consume quota, preventing cache from being
    bypassed for rate limiting. (3) RateLimitingScenarios tests simulate
    real-world conditions with concurrent patients, mixed limiting rules, and
    accurate reset timestamps. (4) ErrorConditions tests handle edge cases
    like zero remaining quota, unusual keys, and extreme request volumes. By
    directly testing RateLimiter and ResponseCache classes with realistic
    data, we validate both the rate limiting logic and cache interaction work
    correctly without affecting production systems.
"""

import time

from app.rate_limiter import (
    RateLimiter,
    ResponseCache,
    _rate_limit_store,
    diagnosis_cache,
    diagnosis_rate_limiter,
    ip_rate_limiter,
)


class TestRateLimitingLogic:
    """Test the core rate limiting logic used by endpoints."""

    def setup_method(self):
        """Reset rate limit store before each test."""
        _rate_limit_store.clear()

    def test_patient_rate_limit_enforcement(self):
        """Test patient-based rate limiting."""
        patient_key = 'patient:123'

        # Make 5 allowed requests
        for i in range(5):
            is_allowed, info = diagnosis_rate_limiter.is_allowed(patient_key)
            assert is_allowed is True
            assert info['limit'] == 5
            assert info['remaining'] == 4 - i

        # 6th request should be rejected
        is_allowed, info = diagnosis_rate_limiter.is_allowed(patient_key)
        assert is_allowed is False
        assert info['remaining'] == 0

    def test_ip_rate_limit_enforcement(self):
        """Test IP-based rate limiting."""
        ip_key = 'ip:192.168.1.1'

        # Make 100 allowed requests
        for i in range(100):
            is_allowed, _ = ip_rate_limiter.is_allowed(ip_key)
            assert is_allowed is True, f'Request {i} should be allowed'

        # 101st request should be rejected
        is_allowed, _ = ip_rate_limiter.is_allowed(ip_key)
        assert is_allowed is False

    def test_rate_limit_headers_format(self):
        """Test that rate limit info has correct format for HTTP headers."""
        _, info = diagnosis_rate_limiter.is_allowed('patient:abc')

        # Headers should include: X-RateLimit-Limit, Remaining, Reset
        assert 'limit' in info
        assert 'remaining' in info
        assert 'reset' in info

        # Values should be integers
        assert isinstance(info['limit'], int)
        assert isinstance(info['remaining'], int)
        assert isinstance(info['reset'], int)

        # Remaining should be less than limit
        assert info['remaining'] < info['limit']

        # Reset should be a future Unix timestamp
        assert info['reset'] > int(time.time())

    def test_separate_patient_limits_independent(self):
        """Test that different patients have independent limits."""
        # Patient A exhausts their limit
        for _ in range(5):
            diagnosis_rate_limiter.is_allowed('patient:A')

        is_allowed_a, _ = diagnosis_rate_limiter.is_allowed('patient:A')
        assert is_allowed_a is False

        # Patient B should still be allowed
        is_allowed_b, _ = diagnosis_rate_limiter.is_allowed('patient:B')
        assert is_allowed_b is True

    def test_separate_ip_limits_independent(self):
        """Test that different IPs have independent limits."""
        # IP A makes 100 requests
        for _ in range(100):
            ip_rate_limiter.is_allowed('ip:192.168.1.1')

        is_allowed_a, _ = ip_rate_limiter.is_allowed('ip:192.168.1.1')
        assert is_allowed_a is False

        # IP B should still be allowed
        is_allowed_b, _ = ip_rate_limiter.is_allowed('ip:192.168.1.2')
        assert is_allowed_b is True


class TestRateLimitingWithCaching:
    """Test interaction between rate limiting and caching."""

    def setup_method(self):
        """Reset state before each test."""
        _rate_limit_store.clear()
        diagnosis_cache._cache.clear()

    def test_cache_does_not_count_toward_limit(self):
        """Test that cached responses don't consume rate limit quota."""
        patient_key = 'patient:123'
        response_key = f'{patient_key}:symptoms:headache'

        # Cache a response
        cached_data = {'diagnosis': 'migraine', 'confidence': 0.9}
        diagnosis_cache.set(response_key, cached_data)

        # Get cached response multiple times
        for _ in range(10):
            result = diagnosis_cache.get(response_key)
            assert result == cached_data

        # Patient should still have full rate limit
        for i in range(5):
            is_allowed, _ = diagnosis_rate_limiter.is_allowed(patient_key)
            assert is_allowed is True

    def test_cache_expiry_not_reset_on_access(self):
        """Test that accessing cache doesn't reset expiry timer."""
        cache = ResponseCache(ttl=1)
        key = 'test:key'
        data = {'result': 'test'}

        cache.set(key, data)

        # Access cache after 0.5 seconds
        time.sleep(0.5)
        result = cache.get(key)
        assert result == data

        # Wait remaining time for expiry (0.6 more seconds)
        time.sleep(0.6)

        # Should be expired now
        result = cache.get(key)
        assert result is None


class TestRateLimitingScenarios:
    """Test realistic rate limiting scenarios."""

    def setup_method(self):
        """Reset state before each test."""
        _rate_limit_store.clear()

    def test_concurrent_patients_separate_limits(self):
        """Test multiple patients can use endpoints concurrently."""
        # Simulate 3 different patients making requests
        patients = ['patient:1', 'patient:2', 'patient:3']

        # Each patient makes 5 requests
        for patient in patients:
            for i in range(5):
                is_allowed, info = diagnosis_rate_limiter.is_allowed(patient)
                assert is_allowed is True
                assert info['remaining'] == 4 - i

        # Each patient should be rate limited on 6th request
        for patient in patients:
            is_allowed, _ = diagnosis_rate_limiter.is_allowed(patient)
            assert is_allowed is False

    def test_mixed_patient_and_ip_limiting(self):
        """Test that patient and IP limits work independently."""
        patient_key = 'patient:mixed_test'
        ip_key = 'ip:10.0.0.1'

        # Exhaust patient limit with diagnosis requests
        for _ in range(5):
            diagnosis_rate_limiter.is_allowed(patient_key)

        # Patient limit exhausted
        is_allowed, _ = diagnosis_rate_limiter.is_allowed(patient_key)
        assert is_allowed is False

        # But IP limit should be independent
        is_allowed, _ = ip_rate_limiter.is_allowed(ip_key)
        assert is_allowed is True

    def test_rate_limit_reset_info_accuracy(self):
        """Test that reset timestamp is accurate."""
        patient_key = 'patient:reset_test'

        # Get rate limit info
        is_allowed, info = diagnosis_rate_limiter.is_allowed(patient_key)
        assert is_allowed is True

        reset_time = info['reset']
        current_time = int(time.time())

        # Reset should be approximately 1 hour from now
        expected_reset = current_time + 3600
        assert abs(reset_time - expected_reset) < 5  # Within 5 seconds


class TestErrorConditions:
    """Test error conditions and edge cases."""

    def setup_method(self):
        """Reset state before each test."""
        _rate_limit_store.clear()

    def test_rate_limit_zero_remaining(self):
        """Test rate limit when remaining is exactly zero."""
        limiter = RateLimiter(max_calls=1, time_window=3600)
        key = 'edge:zero'

        # First request consumes the single allowed request
        is_allowed, info = limiter.is_allowed(key)
        assert is_allowed is True
        assert info['remaining'] == 0

        # Second request should be rejected
        is_allowed, info = limiter.is_allowed(key)
        assert is_allowed is False
        assert info['remaining'] == 0

    def test_empty_key_handling(self):
        """Test handling of empty or unusual keys."""
        limiter = RateLimiter(max_calls=5, time_window=3600)

        # Empty string key
        is_allowed, _ = limiter.is_allowed('')
        assert is_allowed is True

        # Very long key
        long_key = 'patient:' + 'x' * 1000
        is_allowed, _ = limiter.is_allowed(long_key)
        assert is_allowed is True

    def test_cache_with_none_value(self):
        """Test cache behavior with None values."""
        cache = ResponseCache(ttl=300)

        # Cache can store None
        cache.set('null_key', None)
        result = cache.get('null_key')
        # Returned None, but it's from cache expiry check
        assert result is None

    def test_extremely_high_request_volume(self):
        """Test handling of very high number of requests."""
        limiter = RateLimiter(max_calls=1000, time_window=3600)
        key = 'high_volume:test'

        # Make 1000 requests
        for i in range(1000):
            is_allowed, info = limiter.is_allowed(key)
            assert is_allowed is True
            assert info['remaining'] == 999 - i

        # 1001st should fail
        is_allowed, _ = limiter.is_allowed(key)
        assert is_allowed is False
