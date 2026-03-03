import asyncio
import json
import os
import random
import re
import time

import aiohttp
import chardet


# --- Utilities ---
def parse_snippet(snippet: str) -> list[str]:
    segments = snippet.split("...")
    return [s.strip() for s in segments if len(s.strip().split()) > 5]


def sanitize_search_query(query: str) -> str:
    # Remove or replace special characters that might cause issues.
    # This is a basic example; you might need to add more characters or patterns.
    sanitized_query = re.sub(r"[^\w\s]", " ", query)  # Replace non-alphanumeric and non-whitespace with spaces.
    sanitized_query = re.sub(
        r"[\t\r\f\v\n]", " ", sanitized_query
    )  # replace tab, return, formfeed, vertical tab with spaces.
    sanitized_query = re.sub(
        r"\s+", " ", sanitized_query
    ).strip()  # remove duplicate spaces, and trailing/leading spaces.

    return sanitized_query


def filter_links(search_results: list[dict]) -> list[str]:
    links = []
    for result in search_results:
        for item in result.get("items", []):
            if "mime" in item:
                continue
            ext = os.path.splitext(item["link"])[1]
            if ext in ["", ".html", ".htm", ".shtml"]:
                links.append(item["link"])
    return links


async def fetch(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> str:
    if url == "":
        return ""
    user_agents = [
        "Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P)...",
        "Mozilla/5.0 AppleWebKit/537.36...",
        "Mozilla/5.0 (compatible; Googlebot/2.1; +https://www.google.com/bot.html)",
    ]
    headers = {"User-Agent": random.choice(user_agents)}

    async with semaphore:
        try:
            async with session.get(url, headers=headers) as response:
                raw = await response.read()
                detected = chardet.detect(raw)
                encoding = detected["encoding"] or "utf-8"
                return raw.decode(encoding, errors="ignore")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return ""


async def fetch_all(urls: list[str], limit: int = 8) -> list[str]:
    semaphore = asyncio.Semaphore(limit)
    timeout = aiohttp.ClientTimeout(total=5)
    connector = aiohttp.TCPConnector(limit_per_host=limit, force_close=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [fetch(session, url, semaphore) for url in urls]
        return await asyncio.gather(*tasks)


def collect_context(snippet: str, doc: str) -> str:
    snippets = parse_snippet(snippet)
    ctx_paras = []

    for s in snippets:
        pos = doc.replace("\n", " ").find(s)
        if pos == -1:
            continue
        sta = pos
        while sta > 0 and doc[sta] != "\n":
            sta -= 1
        end = pos + len(s)
        while end < len(doc) and doc[end] != "\n":
            end += 1
        para = doc[sta:end].strip()
        if para not in ctx_paras:
            ctx_paras.append(para)

    return "\n".join(ctx_paras)


async def google_search(api_key, query, top_k=5, timeout: int = 60, proxy=None, snippet_only=False) -> list[dict]:
    max_retries = int(os.environ.get("SEARCH_R1_GOOGLE_MAX_RETRIES", "8"))
    backoff_base = float(os.environ.get("SEARCH_R1_GOOGLE_BACKOFF_BASE", "1.5"))
    backoff_cap = float(os.environ.get("SEARCH_R1_GOOGLE_BACKOFF_CAP", "30.0"))
    min_interval = float(os.environ.get("SEARCH_R1_GOOGLE_MIN_INTERVAL", "0.4"))

    # Process-local rate limiter: keep a minimum interval between upstream requests.
    if not hasattr(google_search, "_rate_lock"):
        google_search._rate_lock = asyncio.Lock()
        google_search._next_allowed_ts = 0.0

    async def throttle_once() -> None:
        async with google_search._rate_lock:
            now = time.monotonic()
            wait = google_search._next_allowed_ts - now
            if wait > 0:
                await asyncio.sleep(wait)
            google_search._next_allowed_ts = time.monotonic() + min_interval

    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    session_kwargs = {}
    if proxy:
        session_kwargs["proxy"] = proxy
    async with aiohttp.ClientSession(**session_kwargs) as session:
        response = None
        last_error = None
        for attempt in range(max_retries):
            await throttle_once()
            try:
                async with session.post(
                    "https://google.serper.dev/search",
                    json={
                        "q": query,
                        "num": top_k,
                        "gl": "us",
                        "hl": "en",
                    },
                    headers={
                        "Content-Type": "application/json",
                        "X-API-KEY": api_key,
                    },
                    timeout=timeout_obj,
                ) as resp:
                    if resp.status in (429, 500, 502, 503, 504):
                        retry_after = resp.headers.get("Retry-After", "")
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = min(backoff_cap, backoff_base * (2**attempt))
                        delay += random.uniform(0.0, 0.5)
                        body = await resp.text()
                        last_error = RuntimeError(
                            f"google_search transient HTTP {resp.status}: {body[:200]}"
                        )
                        if attempt + 1 < max_retries:
                            await asyncio.sleep(delay)
                            continue
                        raise last_error

                    resp.raise_for_status()
                    raw = await resp.text()
                    response = json.loads(raw)
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 >= max_retries:
                    raise
                delay = min(backoff_cap, backoff_base * (2**attempt)) + random.uniform(0.0, 0.5)
                await asyncio.sleep(delay)

        if response is None:
            raise RuntimeError(f"google_search failed after {max_retries} retries: {last_error}")

        items = response.get("organic", [])

    contexts = []
    if snippet_only:
        for item in items:
            title = item.get("title", "")
            context = " ".join(parse_snippet(item.get("snippet", "")))
            if title != "" or context != "":
                title = "No title." if not title else title
                context = "No snippet available." if not context else context
                contexts.append(
                    {
                        "document": {"contents": f'"{title}"\n{context}'},
                    }
                )
    else:
        links = [item.get("link", "") for item in items if "link" in item]
        web_contents = await fetch_all(links)
        contexts = []
        for i, item in enumerate(items):
            title = item.get("title", "")
            snippet = item.get("snippet", "")

            context = collect_context(snippet, web_contents[i])
            if title != "" or context != "":
                title = "No title." if not title else title
                context = "No snippet available." if not context else context
                contexts.append(
                    {
                        "document": {"contents": f'"{title}"\n{context}'},
                    }
                )

    return contexts
