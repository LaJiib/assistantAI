"""Utility functions for the Deep Research subagent."""

import asyncio
import logging
from datetime import datetime
from typing import Any, List, Optional
import os
from langchain_core.messages import AIMessage, MessageLikeRepresentation, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from utils.tools import web_search as _web_search_base
from utils.tools import fetch_webpage as _fetch_webpage_base
import httpx
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright





@tool(parse_docstring=True)
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using SearXNG.

    Args:
        query: Search query string.
        max_results: Maximum number of results.
    """
    searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            urljoin(searxng_url, "/search"),
            params={"q": query, "format": "json", "categories": "general"},
        )
        response.raise_for_status()
        data = response.json()

    raw_results = data.get("results", [])
    if not raw_results:
        return "No results found."

    formatted = ["### SEARCH RESULTS"]
    for i, item in enumerate(raw_results[:max_results], 1):
        formatted.append(
            f"### SOURCE [{i}]\n"
            f"**TITLE:** {item.get('title')}\n"
            f"**URL:** {item.get('url')}\n"
            f"**SNIPPET:** {item.get('content')}"
        )
    return "\n\n".join(formatted)


@tool(parse_docstring=True)
async def fetch_webpage(url: str) -> str:
    """Fetch and extract the clean text content of a web page.

    Args:
        url: Full URL to fetch (http or https only).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return f"URL invalide : '{url}'"

    # Réutilise le browser singleton existant
    from utils.tools import _get_browser
    try:
        browser = await _get_browser()
        context = await asyncio.wait_for(browser.new_context(), timeout=5.0)
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
            text = await page.evaluate("""() => {
                ['nav','footer','header','aside','form','script','style','noscript']
                    .forEach(tag => document.querySelectorAll(tag)
                    .forEach(el => el.remove()));
                const main = document.querySelector('main, article, [role="main"]');
                return (main || document.body).textContent
                    .replace(/\\s+/g, ' ')
                    .trim();
            }""")
        finally:
            await context.close()
    except Exception as e:
        return f"Erreur fetch : {e}"

    words = text.split()
    if len(words) > 10000:
        text = " ".join(words[:10000])
    return text or "Contenu vide."








# ---------------------------------------------------------------------------
# Token limit utils
# ---------------------------------------------------------------------------

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Détecte si une exception indique un dépassement de contexte."""
    error_str = str(exception).lower()
    token_keywords = [
        "token", "context", "too long", "maximum", "length",
        "prompt is too long", "reduce"
    ]
    return any(keyword in error_str for keyword in token_keywords)


def get_model_token_limit(model_string: str) -> Optional[int]:
    """Limite de contexte pour Gemma4 via Omlx."""
    return 128000  # Gemma4 26B : 128k tokens


def remove_up_to_last_ai_message(
    messages: list[MessageLikeRepresentation],
) -> list[MessageLikeRepresentation]:
    """Tronque l'historique en supprimant jusqu'au dernier message AI."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]
    return messages


# ---------------------------------------------------------------------------
# Date utils — identique au repo
# ---------------------------------------------------------------------------

def get_today_str() -> str:
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"