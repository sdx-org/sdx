"""Rate limiting utilities for AI endpoints."""

import time

from typing import Optional

# In-memory rate limit store (for single-instance deployment)
# In production, replace with Redis
_rate_limit_store = {}


class RateLimiter:
    """Simple rate limiter for API endpoints."""

    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window

    def is_allowed(self, key: str) -> tuple[bool, dict]:
        """
        Check if request is allowed.

        Args:
            key: Rate limit key (e.g., 'patient:123', 'ip:192.168.1.1')

        Returns
        -------
            (is_allowed, rate_limit_info)
        """
        now = time.time()

        if key not in _rate_limit_store:
            _rate_limit_store[key] = {
                'count': 1,
                'reset_at': now + self.time_window,
            }
            return True, {
                'limit': self.max_calls,
                'remaining': self.max_calls - 1,
                'reset': int(now + self.time_window),
            }

        entry = _rate_limit_store[key]

        # Reset if window expired
        if now >= entry['reset_at']:
            entry['count'] = 1
            entry['reset_at'] = now + self.time_window
            return True, {
                'limit': self.max_calls,
                'remaining': self.max_calls - 1,
                'reset': int(entry['reset_at']),
            }

        # Check if limit exceeded
        if entry['count'] >= self.max_calls:
            return False, {
                'limit': self.max_calls,
                'remaining': 0,
                'reset': int(entry['reset_at']),
            }

        # Increment counter
        entry['count'] += 1

        return True, {
            'limit': self.max_calls,
            'remaining': self.max_calls - entry['count'],
            'reset': int(entry['reset_at']),
        }


class ResponseCache:
    """Simple response cache for AI endpoints."""

    def __init__(self, ttl: int = 300):
        """
        Initialize cache.

        Args:
            ttl: Time to live in seconds (default: 5 minutes)
        """
        self.ttl = ttl
        self._cache = {}

    def get(self, key: str) -> Optional[dict]:
        """Get cached response."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if time.time() >= entry['expires_at']:
            del self._cache[key]
            return None

        return entry['data']

    def set(self, key: str, value: dict) -> None:
        """Set cached response."""
        self._cache[key] = {
            'data': value,
            'expires_at': time.time() + self.ttl,
        }

    def clear_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [
            k for k, v in self._cache.items() if now >= v['expires_at']
        ]
        for k in expired_keys:
            del self._cache[k]


# Global instances
diagnosis_rate_limiter = RateLimiter(
    max_calls=5, time_window=3600
)  # 5 per hour per patient
exam_rate_limiter = RateLimiter(
    max_calls=5, time_window=3600
)  # 5 per hour per patient
ip_rate_limiter = RateLimiter(
    max_calls=100, time_window=3600
)  # 100 per hour per IP

diagnosis_cache = ResponseCache(ttl=300)  # 5 minute cache
exam_cache = ResponseCache(ttl=300)  # 5 minute cache
