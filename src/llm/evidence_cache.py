"""
src/llm/evidence_cache.py
─────────────────────────
Disk cache + rate limiter + retry for the live evidence APIs (NCBI E-utilities,
NIH DailyMed).

Why this exists:
* ``rag_corpus`` created ``self.cache_dir`` and never read or wrote it, so every
  query re-hit NCBI. That is slow, non-reproducible, and rate-limit fragile.
* NCBI permits 3 requests/second without an API key and 10/second with one.
  Exceeding it returns HTTP 429 and, sustained, gets the caller blocked.
* A network outage previously surfaced as the same "integrity check failed"
  record as a genuine evidence rejection. Callers must be able to tell them apart,
  so transport failure raises :class:`RetrievalUnavailable` instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["RetrievalUnavailable", "EvidenceCache", "get_default_cache"]

NCBI_API_KEY_ENV = "NCBI_API_KEY"


class RetrievalUnavailable(RuntimeError):
    """Transport-level failure: network down, timeout, rate limited, 5xx.

    Explicitly *not* an evidence-quality judgement. Callers must surface this
    as "evidence could not be retrieved", never as "no evidence exists".
    """


@dataclass
class _Entry:
    body: str
    fetched_at: float


class _RateLimiter:
    """Simple token-spacing limiter, safe across threads."""

    def __init__(self, per_second: float) -> None:
        self._min_interval = 1.0 / max(per_second, 0.1)
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


class EvidenceCache:
    """Content-addressed disk cache for HTTP GETs, with TTL and retry."""

    def __init__(
        self,
        cache_dir: os.PathLike | str,
        ttl_seconds: int = 7 * 24 * 3600,
        requests_per_second: Optional[float] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self.timeout = timeout
        self.max_retries = max_retries
        self.offline = offline
        has_key = bool(os.environ.get(NCBI_API_KEY_ENV))
        self.limiter = _RateLimiter(requests_per_second or (9.0 if has_key else 2.5))
        self.stats = {"hit": 0, "miss": 0, "error": 0, "stale_served": 0}

    # ── internals ────────────────────────────────────────────────────────
    def _path_for(self, url: str) -> Path:
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{h}.json"

    def _read(self, url: str) -> Optional[_Entry]:
        p = self._path_for(url)
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return _Entry(body=raw["body"], fetched_at=float(raw["fetched_at"]))
        except Exception:
            return None

    def _write(self, url: str, body: str) -> None:
        try:
            self._path_for(url).write_text(
                json.dumps({"url": url, "body": body, "fetched_at": time.time()}),
                encoding="utf-8",
            )
        except Exception:
            pass  # cache write failure must never break retrieval

    @staticmethod
    def _with_api_key(url: str) -> str:
        key = os.environ.get(NCBI_API_KEY_ENV)
        if key and "eutils.ncbi.nlm.nih.gov" in url and "api_key=" not in url:
            return url + ("&" if "?" in url else "?") + f"api_key={key}"
        return url

    # ── public ───────────────────────────────────────────────────────────
    def get(self, url: str, *, force_refresh: bool = False) -> str:
        """
        Fetch ``url``, preferring a fresh cache entry.

        Raises
        ------
        RetrievalUnavailable
            On any transport failure with no usable cache entry.
        """
        entry = None if force_refresh else self._read(url)
        if entry is not None and (time.time() - entry.fetched_at) < self.ttl:
            self.stats["hit"] += 1
            return entry.body

        if self.offline:
            if entry is not None:
                self.stats["stale_served"] += 1
                return entry.body
            self.stats["error"] += 1
            raise RetrievalUnavailable(f"offline mode and no cached copy of {url}")

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                self.limiter.acquire()
                req = urllib.request.Request(
                    self._with_api_key(url),
                    headers={"User-Agent": "ClinicalDigitalTwin/1.0 (research)"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                self._write(url, body)
                self.stats["miss"] += 1
                return body
            except urllib.error.HTTPError as e:      # noqa: PERF203
                last_err = e
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep((2 ** attempt) * 0.5 + random.random() * 0.3)
                    continue
                break
            except Exception as e:
                last_err = e
                time.sleep((2 ** attempt) * 0.4 + random.random() * 0.2)

        # transport failed — serve a stale entry if we have one, else signal clearly
        if entry is not None:
            self.stats["stale_served"] += 1
            return entry.body
        self.stats["error"] += 1
        raise RetrievalUnavailable(f"could not retrieve {url}: {last_err}")

    def get_json(self, url: str, *, force_refresh: bool = False) -> dict:
        body = self.get(url, force_refresh=force_refresh)
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise RetrievalUnavailable(f"malformed JSON from {url}: {e}") from e


_DEFAULT: Optional[EvidenceCache] = None


def get_default_cache(cache_dir: os.PathLike | str | None = None) -> EvidenceCache:
    """Process-wide cache instance (created on first use)."""
    global _DEFAULT
    if _DEFAULT is None:
        root = Path(__file__).resolve().parents[2]
        _DEFAULT = EvidenceCache(
            cache_dir or (root / "data" / "processed" / "ncbi_cache"),
            offline=os.environ.get("CDT_OFFLINE", "").lower() in ("1", "true", "yes"),
        )
    return _DEFAULT
