"""Polite HTTP for the public stats sites: one session, a spacing between
requests to the same host, and retries with backoff on transient errors."""

from __future__ import annotations

import time
from urllib.parse import urlsplit

import requests

USER_AGENT = "npl-sim data pipeline (+https://github.com/dwolfmeyer/npl-baseball-sim)"
RETRY_STATUS = {429, 500, 502, 503, 504}


class Http:
    def __init__(self, min_interval: float = 1.0, retries: int = 4, timeout: float = 60.0):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.min_interval = min_interval
        self.retries = retries
        self.timeout = timeout
        self._last: dict[str, float] = {}

    def _wait(self, host: str) -> None:
        gap = time.monotonic() - self._last.get(host, 0.0)
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last[host] = time.monotonic()

    def get(self, url: str, params: dict | None = None) -> bytes:
        host = urlsplit(url).netloc
        for attempt in range(self.retries + 1):
            self._wait(host)
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.retries:
                    raise
            else:
                if resp.status_code not in RETRY_STATUS or attempt == self.retries:
                    resp.raise_for_status()
                    return resp.content
            time.sleep(2 ** (attempt + 1))
        raise AssertionError("unreachable")
