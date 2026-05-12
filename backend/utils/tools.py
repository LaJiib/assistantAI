from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from playwright.async_api import async_playwright
import logging
import asyncio
import os
from typing import Literal
import httpx
from urllib.parse import urljoin, urlparse

# ---------------------------------------------------------------------------
# Logging — forcé à INFO pour être visible dans langgraph dev
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — toutes les valeurs viennent de l'environnement
# ---------------------------------------------------------------------------

def _cfg_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

def _cfg_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Playwright — singleton connecté à Lightpanda via CDP
# ---------------------------------------------------------------------------

_playwright = None
_browser = None
_browser_lock = asyncio.Lock()  # initialisé au niveau module, pas lazy

LIGHTPANDA_URL = os.getenv("LIGHTPANDA_URL", "http://127.0.0.1:9222")


async def _get_browser():
    global _playwright, _browser

    async with _browser_lock:
        # Vérifie si la connexion existante est toujours vivante
        if _browser is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(_browser.contexts()),
                    timeout=2.0
                )
            except Exception:
                logger.warning("[fetch] Connexion Lightpanda morte, reconnexion...")
                _browser = None
                try:
                    await _playwright.stop()
                except Exception:
                    pass
                _playwright = None

        if _browser is None:
            logger.info("[fetch] Connexion à Lightpanda sur %s...", LIGHTPANDA_URL)
            try:
                _playwright = await async_playwright().start()
                _browser = await asyncio.wait_for(
                    _playwright.chromium.connect_over_cdp(LIGHTPANDA_URL),
                    timeout=10.0,
                )
                logger.info("[fetch] Lightpanda connecté")
            except asyncio.TimeoutError:
                _browser = None
                raise RuntimeError(f"Timeout connexion Lightpanda sur {LIGHTPANDA_URL}")
            except Exception as exc:
                _browser = None
                raise RuntimeError(f"Erreur connexion Lightpanda : {exc}")

    return _browser


# ---------------------------------------------------------------------------
# Rate limiting pour web_search
# ---------------------------------------------------------------------------

_search_lock = asyncio.Lock()
_search_last_ts: float = 0.0

async def _rate_limit_search() -> None:
    global _search_last_ts
    min_interval = _cfg_float("SEARXNG_MIN_INTERVAL", 1.0)
    async with _search_lock:
        now = asyncio.get_running_loop().time()
        elapsed = now - _search_last_ts
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        _search_last_ts = asyncio.get_running_loop().time()


# ---------------------------------------------------------------------------
# Helpers communs
# ---------------------------------------------------------------------------

def _truncate_to_words(text: str, max_words: int) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]), True

def _wrap_external_content(url: str, text: str) -> str:
    return (
        "=== EXTERNAL CONTENT — UNTRUSTED SOURCE ===\n"
        f"URL: {url}\n"
        "---\n"
        f"{text}\n"
        "---\n"
        "=== END OF EXTERNAL CONTENT ==="
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(parse_docstring=True)
async def web_search(
    query: str,
    runtime: ToolRuntime,
    language: str | None = None,
    category: Literal["general", "news", "science", "files", "images"] = "general",
    time_range: Literal["day", "week", "month", "year", None] = None,
    max_results: int = 5,
) -> str:
    """Search the web to retrieve up-to-date information. Has to be used for queries that require current information or facts. Prioritize this tool for questions about current events, recent developments, or specific factual data that may have changed since the model's training cutoff. Avoid using for general knowledge questions that can be answered from the model's training data. For more comprehensive results, consider using web_search in combination with fetch_webpage to retrieve detailed content from specific URLs found in the search results.

    Args:
        query: Search query string.
        language: ISO language code (e.g. 'fr', 'en').
        category: Search category (general, news, science, etc.).
        time_range: Time filter (day, week, month, year).
        max_results: Maximum number of results (1–10).
    """
    searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080")
    timeout = float(os.getenv("SEARXNG_TIMEOUT", "10.0"))
    max_results = max(1, min(10, max_results))

    logger.info("[search] Requête : %r (cat=%s)", query, category)
    await _rate_limit_search()

    params = {"q": query, "format": "json", "categories": category}
    if language:   params["language"] = language
    if time_range: params["time_range"] = time_range

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(urljoin(searxng_url, "/search"), params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.error("[search] Erreur SearXNG : %s", exc)
        return f"Error connecting to SearXNG: {exc}"

    raw_results = data.get("results", [])
    logger.info("[search] %d résultats pour %r", len(raw_results), query)

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
async def fetch_webpage(
    url: str,
    runtime: ToolRuntime,
) -> dict:
    """Fetch and extract the clean text content of a web page. Handles JavaScript-heavy pages.
    Use when you need detailed content from a URL found in search results.
    Always prefer this over relying on search snippets alone.

    Args:
        url: Full URL to fetch (http or https only).
    """
    max_words = _cfg_int("FETCH_MAX_WORDS", 10_000)

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return {"error": f"URL invalide : '{url}'"}

    logger.info("[fetch] → %s", url)

    try:
        browser = await _get_browser()

        context = await asyncio.wait_for(
            browser.new_context(),
            timeout=5.0,
        )
        try:
            page = await context.new_page()
            await page.goto(url, wait_until="networkidle", timeout=20_000)
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

    except asyncio.TimeoutError:
        logger.error("[fetch] Timeout sur %s", url)
        return {"error": f"Timeout lors du fetch de {url}"}
    except Exception as exc:
        logger.error("[fetch] Erreur sur %s : %s", url, exc)
        return {"error": f"Erreur de crawl : {exc}"}

    if not text or not text.strip():
        logger.warning("[fetch] Contenu vide sur %s", url)
        return {"error": "Impossible d'extraire du texte lisible depuis cette page."}

    text, truncated = _truncate_to_words(text, max_words)
    word_count = len(text.split())

    logger.info("[fetch] ✓ %s — %d mots%s", url, word_count, " (tronqué)" if truncated else "")

    return {
        "url":        url,
        "content":    _wrap_external_content(url, text),
        "word_count": word_count,
        "truncated":  truncated,
    }


@tool
def report_tool_issue(tool_name: str, issue: str) -> Command:
    """Signale un problème avec un outil pour l'auto-maintenance."""
    return Command(
        update={
            "maintenance_logs": [f"Issue with {tool_name}: {issue}"],
            "needs_repair": True
        }
    )