import asyncio
import ipaddress
import json
import os
import time
from typing import Any
from urllib.parse import urlparse

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    text = host.strip().lower()
    if text == "localhost":
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def _is_loopback_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return _is_loopback_host(parsed.hostname)


UPSTREAM_URL = os.environ.get("LOCAL_SEARCH_UPSTREAM_URL", "http://127.0.0.1:8000/retrieve")
UPSTREAM_TIMEOUT = _env_float("LOCAL_SEARCH_UPSTREAM_TIMEOUT", 5.0)
RETRY_TIMES = _env_int("LOCAL_SEARCH_RETRY_TIMES", 1)
RETRY_BACKOFF = _env_float("LOCAL_SEARCH_RETRY_BACKOFF", 0.2)
MAX_INFLIGHT = max(1, _env_int("LOCAL_SEARCH_MAX_INFLIGHT", 128))
QUEUE_WAIT_TIMEOUT = _env_float("LOCAL_SEARCH_QUEUE_WAIT_TIMEOUT", 1.5)
CONNECTOR_LIMIT = max(1, _env_int("LOCAL_SEARCH_CONNECTOR_LIMIT", 512))
CONNECTOR_PER_HOST = max(1, _env_int("LOCAL_SEARCH_CONNECTOR_PER_HOST", 256))
DEFAULT_TOPK = max(1, _env_int("LOCAL_SEARCH_DEFAULT_TOPK", 3))
TRUST_ENV = _env_bool("LOCAL_SEARCH_TRUST_ENV", False)
BYPASS_PROXY_FOR_LOCAL_UPSTREAM = _env_bool("LOCAL_SEARCH_BYPASS_PROXY_FOR_LOCAL_UPSTREAM", True)


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(DEFAULT_TOPK, ge=1, le=50)
    timeout: float = Field(UPSTREAM_TIMEOUT, gt=0.0, le=120.0)
    proxy: str | None = Field(None, description="Optional http proxy for upstream call")


class _GatewayState:
    def __init__(self) -> None:
        self.session: aiohttp.ClientSession | None = None
        self.semaphore = asyncio.Semaphore(MAX_INFLIGHT)
        self.lock = asyncio.Lock()
        self.total_requests = 0
        self.success_requests = 0
        self.busy_requests = 0
        self.timeout_requests = 0
        self.upstream_error_requests = 0
        self.in_flight = 0
        self.max_in_flight_seen = 0
        self.success_latency_ms_sum = 0.0

    async def on_request_start(self) -> None:
        async with self.lock:
            self.total_requests += 1
            self.in_flight += 1
            if self.in_flight > self.max_in_flight_seen:
                self.max_in_flight_seen = self.in_flight

    async def on_request_end(self) -> None:
        async with self.lock:
            self.in_flight = max(0, self.in_flight - 1)

    async def on_success(self, latency_ms: float) -> None:
        async with self.lock:
            self.success_requests += 1
            self.success_latency_ms_sum += latency_ms

    async def on_busy(self) -> None:
        async with self.lock:
            self.busy_requests += 1

    async def on_timeout(self) -> None:
        async with self.lock:
            self.timeout_requests += 1

    async def on_upstream_error(self) -> None:
        async with self.lock:
            self.upstream_error_requests += 1

    async def snapshot(self) -> dict[str, Any]:
        async with self.lock:
            avg = (
                self.success_latency_ms_sum / self.success_requests
                if self.success_requests > 0
                else 0.0
            )
            return {
                "status": "ok",
                "upstream_url": UPSTREAM_URL,
                "trust_env": TRUST_ENV,
                "bypass_proxy_for_local_upstream": BYPASS_PROXY_FOR_LOCAL_UPSTREAM,
                "max_inflight": MAX_INFLIGHT,
                "queue_wait_timeout": QUEUE_WAIT_TIMEOUT,
                "connector_limit": CONNECTOR_LIMIT,
                "connector_per_host": CONNECTOR_PER_HOST,
                "retry_times": RETRY_TIMES,
                "retry_backoff": RETRY_BACKOFF,
                "upstream_timeout_default": UPSTREAM_TIMEOUT,
                "total_requests": self.total_requests,
                "success_requests": self.success_requests,
                "busy_requests": self.busy_requests,
                "timeout_requests": self.timeout_requests,
                "upstream_error_requests": self.upstream_error_requests,
                "in_flight": self.in_flight,
                "max_in_flight_seen": self.max_in_flight_seen,
                "avg_success_latency_ms": round(avg, 3),
            }


STATE = _GatewayState()
app = FastAPI(title="Local Search Gateway", version="1.0.0")


def _extract_retrieve_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result", [])
    if not isinstance(result, list) or not result:
        return []
    first = result[0]
    if not isinstance(first, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in first:
        if isinstance(item, dict):
            if "document" in item:
                normalized.append(item)
            else:
                normalized.append({"document": item})
    return normalized


async def _ensure_session() -> aiohttp.ClientSession:
    if STATE.session is None or STATE.session.closed:
        connector = aiohttp.TCPConnector(limit=CONNECTOR_LIMIT, limit_per_host=CONNECTOR_PER_HOST)
        timeout = aiohttp.ClientTimeout(total=None)
        # Default to NOT inheriting process proxy env to keep local upstream (127.0.0.1) stable.
        STATE.session = aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=TRUST_ENV)
    return STATE.session


async def _search_via_retrieval_server(
    upstream_url: str,
    query: str,
    top_k: int,
    timeout_sec: float,
    proxy: str | None,
) -> list[dict[str, Any]]:
    session = await _ensure_session()
    payload = {"queries": [query], "topk": top_k, "return_scores": False}
    effective_proxy = proxy
    if BYPASS_PROXY_FOR_LOCAL_UPSTREAM and _is_loopback_url(upstream_url):
        effective_proxy = None

    total_attempts = max(1, RETRY_TIMES + 1)
    last_err: Exception | None = None
    for attempt in range(1, total_attempts + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_sec)
            async with session.post(upstream_url, json=payload, timeout=timeout, proxy=effective_proxy) as resp:
                text = await resp.text()
                if resp.status in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"upstream_http_{resp.status}: {text[:500]}")
                resp.raise_for_status()
                data = json.loads(text)
                return _extract_retrieve_results(data)
        except asyncio.TimeoutError as exc:
            last_err = exc
            if attempt < total_attempts:
                await asyncio.sleep(RETRY_BACKOFF * attempt)
                continue
            raise
        except Exception as exc:
            last_err = exc
            if attempt < total_attempts:
                await asyncio.sleep(RETRY_BACKOFF * attempt)
                continue
            raise RuntimeError(str(last_err) if last_err is not None else "upstream_error")

    raise RuntimeError("upstream_unreachable")


async def local_search(
    search_url: str,
    query: str,
    top_k: int = DEFAULT_TOPK,
    timeout: float = UPSTREAM_TIMEOUT,
    proxy: str | None = None,
) -> list[dict[str, Any]]:
    if search_url.rstrip("/").endswith("/search"):
        connector = aiohttp.TCPConnector(limit=CONNECTOR_LIMIT, limit_per_host=CONNECTOR_PER_HOST)
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout_obj, trust_env=TRUST_ENV) as session:
            payload = {"query": query, "top_k": top_k, "timeout": timeout, "proxy": proxy}
            async with session.post(search_url, json=payload, proxy=proxy) as resp:
                resp.raise_for_status()
                data = await resp.json()
                results = data.get("results", [])
                if isinstance(results, list):
                    return results
                return []

    return await _search_via_retrieval_server(
        upstream_url=search_url,
        query=query,
        top_k=top_k,
        timeout_sec=timeout,
        proxy=proxy,
    )


@app.on_event("startup")
async def _startup() -> None:
    await _ensure_session()


@app.on_event("shutdown")
async def _shutdown() -> None:
    if STATE.session is not None and not STATE.session.closed:
        await STATE.session.close()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
async def stats() -> dict[str, Any]:
    return await STATE.snapshot()


@app.post("/search")
async def search_endpoint(req: SearchRequest) -> dict[str, Any]:
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is empty")

    try:
        await asyncio.wait_for(STATE.semaphore.acquire(), timeout=QUEUE_WAIT_TIMEOUT)
    except asyncio.TimeoutError:
        await STATE.on_busy()
        raise HTTPException(status_code=429, detail="gateway busy: queue wait timeout exceeded")

    await STATE.on_request_start()
    started = time.perf_counter()
    try:
        results = await _search_via_retrieval_server(
            upstream_url=UPSTREAM_URL,
            query=query,
            top_k=req.top_k,
            timeout_sec=req.timeout,
            proxy=req.proxy,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        await STATE.on_success(latency_ms)
        return {
            "status": "success",
            "query": query,
            "top_k": req.top_k,
            "count": len(results),
            "results": results,
            "error": None,
            "latency_ms": round(latency_ms, 3),
        }
    except asyncio.TimeoutError:
        await STATE.on_timeout()
        raise HTTPException(status_code=504, detail="upstream timeout")
    except Exception as exc:
        await STATE.on_upstream_error()
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}")
    finally:
        await STATE.on_request_end()
        STATE.semaphore.release()


def _main() -> None:
    host = os.environ.get("LOCAL_SEARCH_API_HOST", "0.0.0.0")
    port = _env_int("LOCAL_SEARCH_API_PORT", 18081)
    workers = max(1, _env_int("LOCAL_SEARCH_API_WORKERS", 1))
    log_level = os.environ.get("LOCAL_SEARCH_LOG_LEVEL", "info")

    if workers == 1:
        uvicorn.run(app, host=host, port=port, log_level=log_level)
        return

    uvicorn.run("local_search_server:app", host=host, port=port, workers=workers, log_level=log_level)


if __name__ == "__main__":
    _main()
