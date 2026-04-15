# tools/builtin/web.py
"""
Web tools — recherche SearXNG et lecture de pages web.

Tools Pydantic AI enregistrés sur l'agent iris (Étape 3).
Tous les paramètres de configuration viennent d'os.environ (pas de valeurs hardcodées).

Sécurité :
    fetch_webpage : protection SSRF via _SSRFTransport — résolution DNS + vérification
    IP au moment de la connexion, puis pinning de l'IP résolue dans l'URL pour
    éliminer la race condition DNS rebinding.

Rate limiting :
    web_search : intervalle minimum configurable entre requêtes (SEARXNG_MIN_INTERVAL).

Format réponse :
    web_search   → list[dict] (succès) ou {"error": str}
    fetch_webpage → {"url", "content", "word_count", "truncated"} (succès) ou {"error": str}
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import socket
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from pydantic_ai import RunContext, ToolReturn

from core.agent import IrisDeps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — toutes les valeurs viennent de l'environnement
# ---------------------------------------------------------------------------

def _cfg_str(key: str, default: str) -> str:
    return os.environ.get(key, default)

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
# Helpers SSRF
# ---------------------------------------------------------------------------

def _is_blocked_ip(ip_str: str) -> bool:
    """
    Retourne True si l'IP appartient à une plage réservée/privée à bloquer.

    Bloque : loopback, lien-local, privé, multicast, réservé, non-spécifié.
    Retourne True (bloque) si l'adresse ne peut pas être parsée.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # IP invalide → bloquer par défaut

    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _rewrite_url_with_ip(original_url: httpx.URL, resolved_ip: str) -> httpx.URL:
    """
    Réécrit l'URL en substituant le hostname par l'IP résolue.

    IPv6 : encodé avec crochets [::ip] comme exigé par les RFC.
    Cela empêche httpx de refaire une résolution DNS (IP pinning).
    """
    try:
        addr = ipaddress.ip_address(resolved_ip)
        if isinstance(addr, ipaddress.IPv6Address):
            host_in_url = f"[{resolved_ip}]"
        else:
            host_in_url = resolved_ip
    except ValueError:
        host_in_url = resolved_ip

    return original_url.copy_with(host=host_in_url)


# ---------------------------------------------------------------------------
# Transport httpx custom — protection SSRF avec IP pinning
# ---------------------------------------------------------------------------

class _SSRFTransport(httpx.AsyncHTTPTransport):
    """
    Transport httpx qui vérifie les IPs résolues AVANT d'ouvrir la connexion
    et réécrit l'URL avec l'IP pinned pour prévenir le DNS rebinding.

    Workflow pour chaque requête :
      1. Résoudre le hostname via socket.getaddrinfo() dans asyncio.to_thread
      2. Vérifier que chaque IP résolue n'est pas privée/loopback/link-local
      3. Réécrire l'URL avec la première IP sûre (évite toute re-résolution DNS)
      4. Ajouter/écraser le header Host avec le hostname original
      5. Déléguer à httpx.AsyncHTTPTransport.handle_async_request()

    Élimine la race condition DNS rebinding : après le pinning, httpx se connecte
    directement à l'IP vérifiée sans refaire de lookup DNS.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        hostname = request.url.host
        port = request.url.port

        # Résoudre le hostname dans un thread (opération bloquante)
        try:
            infos = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                port,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise ValueError(f"SSRF check: cannot resolve hostname '{hostname}': {exc}") from exc

        if not infos:
            raise ValueError(f"SSRF check: no addresses resolved for '{hostname}'")

        # Vérifier toutes les IPs résolues
        for info in infos:
            ip = info[4][0]
            if _is_blocked_ip(ip):
                raise ValueError(
                    f"SSRF blocked: '{hostname}' resolves to '{ip}' "
                    f"(private/loopback/link-local address)"
                )

        # Pinning : réécrire l'URL avec la première IP sûre
        pinned_ip = infos[0][4][0]
        pinned_url = _rewrite_url_with_ip(request.url, pinned_ip)

        # Reconstruire la requête avec l'URL pinnée et le header Host original.
        # httpx.Request n'a pas de copy_with — on crée une nouvelle instance.
        # Le header Host est forcé au hostname original pour le virtual hosting.
        new_headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        new_headers["host"] = hostname

        pinned_request = httpx.Request(
            method=request.method,
            url=pinned_url,
            headers=new_headers,
        )

        logger.debug(
            "[iris:fetch] SSRF check OK — %s resolved to %s (pinned)",
            hostname,
            pinned_ip,
        )

        return await super().handle_async_request(pinned_request)


# ---------------------------------------------------------------------------
# Rate limiting pour web_search
# ---------------------------------------------------------------------------

_search_lock = asyncio.Lock()
_search_last_ts: float = 0.0


async def _rate_limit_search() -> None:
    """Impose un délai minimum entre deux appels à web_search."""
    global _search_last_ts
    min_interval = _cfg_float("SEARXNG_MIN_INTERVAL", 1.0)
    async with _search_lock:
        loop = asyncio.get_running_loop()
        now = loop.time()
        elapsed = now - _search_last_ts
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        _search_last_ts = asyncio.get_running_loop().time()


# ---------------------------------------------------------------------------
# Tool 1 — web_search
# ---------------------------------------------------------------------------

async def web_search(
    ctx: RunContext[IrisDeps],
    query: str,
    language: str | None = None,
    category: Literal["general", "news", "science", "files", "images"] = "general",
    time_range: Literal["day", "week", "month", "year", None] = None,
    max_results: int = 5,
) -> ToolReturn | dict[str, str]:
    """
    Search the web to retrieve up-to-date information, external knowledge, and verifiable facts across any domain.
    
    STRATEGIC INSTRUCTIONS:
    1. TRIGGERS: Use this tool whenever a query requires precision, recency, or certainty beyond your internal weights (e.g., fact-checking, current events, technical documentation, or specific data).
    2. DEPTH: If the returned snippets are too generic or only describe a website's homepage, identify the most relevant URL and immediately use the 'fetch_webpage' tool to extract the full content.
    3. STRICT FACTUALITY: Base your final answer strictly on the returned snippets. NEVER invent or assume details not present in the search results.
    4. EVALUATION: Use the 'RELEVANCE SCORE' to prioritize highly rated sources when faced with conflicting information.

    Args:
        query: Search query string.
        language: ISO language code (e.g. 'fr', 'en', 'de').
                  If not provided, no language filter is applied.
        category: Search category. One of: 'general', 'news', 'science',
                  'files', 'images'. Defaults to 'general'.
        time_range: Time filter for results. One of: 'day', 'week', 'month',
                    'year'. If not provided, no time filter is applied.
        max_results: Maximum number of results to return (1–10). Defaults to 5.
    """
    searxng_url = _cfg_str("SEARXNG_URL", "http://localhost:8080")
    timeout = _cfg_float("SEARXNG_TIMEOUT", 10.0)

    # Clamper max_results entre 1 et 10
    max_results = max(1, min(10, max_results))

    # Rate limiting
    await _rate_limit_search()

    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "categories": category,
    }
    if language is not None:
        params["language"] = language
    if time_range is not None:
        params["time_range"] = time_range

    logger.info("[iris:web_search] query=%r lang=%s cat=%s range=%s n=%d",
                query, language, category, time_range, max_results)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                urljoin(searxng_url, "/search"),
                params=params,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        return {"error": f"SearXNG timeout after {timeout}s. Is it running on {searxng_url}?"}
    except httpx.ConnectError:
        return {"error": f"Cannot connect to SearXNG at {searxng_url}. Is it running?"}
    except httpx.HTTPStatusError as exc:
        return {"error": f"SearXNG returned HTTP {exc.response.status_code}."}
    except Exception as exc:
        return {"error": f"SearXNG error: {exc}"}

    raw_results: list[dict] = data.get("results", [])
    if not raw_results:
        return {"error": "No results found for this query."}

    results_slice = raw_results[:max_results]

    # Markdown pour le modèle — format Structured Reference Style avec Score
    formatted = ["### SEARCH RESULTS"]
    for i, item in enumerate(results_slice, 1):
        title = item.get("title", "N/A").strip()
        url = item.get("url", "N/A").strip()
        snippet = item.get("content", "No snippet available.").strip()
        score = item.get("score")
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"

        entry = (
            f"### SOURCE [{i}]\n"
            f"**RELEVANCE SCORE:** {score_str}\n"
            f"**TITLE:** {title}\n"
            f"**URL:** {url}\n"
            f"**SNIPPET:** {snippet}"
        )
        formatted.append(entry)

    markdown = "\n\n".join(formatted)

    # Métadonnées structurées pour l'aperçu frontend (titre + URL uniquement)
    preview_results = [
        {
            "title": item.get("title", "N/A").strip(),
            "url": item.get("url", "N/A").strip(),
        }
        for item in results_slice
    ]

    return ToolReturn(
        return_value=markdown,
        metadata={
            "type": "searchResults",
            "query": query,
            "count": len(results_slice),
            "results": preview_results,
        },
    )


# ---------------------------------------------------------------------------
# Tool 2 — fetch_webpage
# ---------------------------------------------------------------------------

def _truncate_to_words(text: str, max_words: int) -> tuple[str, bool]:
    """Retourne (texte tronqué, truncated_bool)."""
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]), True


def _wrap_external_content(url: str, text: str) -> str:
    """Encapsule le contenu externe pour signaler au modèle qu'il est non fiable."""
    return (
        "=== EXTERNAL CONTENT — UNTRUSTED SOURCE ===\n"
        f"URL: {url}\n"
        "---\n"
        f"{text}\n"
        "---\n"
        "=== END OF EXTERNAL CONTENT ==="
    )


async def fetch_webpage(
    ctx: RunContext[IrisDeps],
    url: str,
    timeout: float = 10.0,
) -> ToolReturn | dict[str, str]:
    """
    Fetch and extract the clean text content of a web page.

    Returns plain text with no HTML markup. Private/internal IP ranges are
    blocked before any connection attempt (SSRF protection). The returned
    content is clearly marked as an untrusted external source to mitigate
    prompt injection risks.

    Args:
        url: Full URL to fetch (http or https only).
        timeout: Request timeout in seconds. Defaults to 10.0, maximum 30.0.
    """
    fetch_timeout = min(_cfg_float("FETCH_TIMEOUT", 10.0), 30.0, timeout)
    max_bytes = _cfg_int("FETCH_MAX_BYTES", 5 * 1024 * 1024)  # 5 MB
    max_words = _cfg_int("FETCH_MAX_WORDS", 10_000)
    max_words_fallback = _cfg_int("FETCH_MAX_WORDS_FALLBACK", 3_000)

    # ── 1. Validation du schéma ───────────────────────────────────────────
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"error": f"Only http/https URLs are supported (got '{parsed.scheme}')."}
    if not parsed.hostname:
        return {"error": "Invalid URL: missing hostname."}

    transport = _SSRFTransport()

    try:
        async with httpx.AsyncClient(transport=transport, timeout=fetch_timeout, follow_redirects=True) as client:

            # ── 2. HEAD request — vérification Content-Type et taille ────
            try:
                head_resp = await client.head(url)
                head_resp.raise_for_status()
            except ValueError as exc:
                # SSRF block levé par le transport
                return {"error": str(exc)}
            except httpx.TimeoutException:
                return {"error": f"HEAD request timed out after {fetch_timeout}s."}
            except httpx.HTTPStatusError as exc:
                return {"error": f"HTTP {exc.response.status_code} on HEAD request."}
            except httpx.RequestError as exc:
                return {"error": f"Network error on HEAD request: {exc}"}

            content_type = head_resp.headers.get("content-type", "")
            if content_type and not content_type.lower().startswith("text/"):
                return {
                    "error": (
                        f"Unsupported content type '{content_type}'. "
                        "Only text/* content types are accepted."
                    )
                }

            content_length = head_resp.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > max_bytes:
                        return {
                            "error": (
                                f"Page too large ({int(content_length):,} bytes). "
                                f"Maximum allowed: {max_bytes:,} bytes."
                            )
                        }
                except ValueError:
                    pass  # Content-Length invalide → on continue en streaming

            # ── 3. GET request — lecture limitée à max_bytes ─────────────
            try:
                async with client.stream("GET", url) as get_resp:
                    get_resp.raise_for_status()

                    # Vérifier le Content-Type sur la réponse GET si absent du HEAD
                    if not content_type:
                        ct = get_resp.headers.get("content-type", "")
                        if ct and not ct.lower().startswith("text/"):
                            return {
                                "error": (
                                    f"Unsupported content type '{ct}'. "
                                    "Only text/* content types are accepted."
                                )
                            }

                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in get_resp.aiter_bytes(chunk_size=65536):
                        total += len(chunk)
                        if total > max_bytes:
                            return {
                                "error": (
                                    f"Page exceeds size limit of {max_bytes:,} bytes "
                                    "during download."
                                )
                            }
                        chunks.append(chunk)

            except ValueError as exc:
                return {"error": str(exc)}
            except httpx.TimeoutException:
                return {"error": f"GET request timed out after {fetch_timeout}s."}
            except httpx.HTTPStatusError as exc:
                return {"error": f"HTTP {exc.response.status_code} fetching page."}
            except httpx.RequestError as exc:
                return {"error": f"Network error fetching page: {exc}"}

    except Exception as exc:
        return {"error": f"Unexpected error: {exc}"}

    html = b"".join(chunks).decode("utf-8", errors="replace")

    # ── 4. Extraction texte avec trafilatura ─────────────────────────────
    text = trafilatura.extract(html, include_links=False, include_images=False)

    using_fallback = False
    if text is None:
        logger.debug("[iris:fetch] trafilatura primary failed — trying favor_recall")
        text = trafilatura.extract(html, favor_recall=True, include_links=False, include_images=False)
        using_fallback = True

    if not text:
        return {"error": "Could not extract readable text content from this page."}

    # ── 5. Troncature ────────────────────────────────────────────────────
    word_limit = max_words_fallback if using_fallback else max_words
    text, truncated = _truncate_to_words(text, word_limit)

    word_count = len(text.split())

    logger.info(
        "[iris:fetch] %s — %d mots%s",
        url,
        word_count,
        " (tronqué)" if truncated else "",
    )

    return ToolReturn(
        return_value={
            "url":        url,
            "content":    _wrap_external_content(url, text),
            "word_count": word_count,
            "truncated":  truncated,
        },
        metadata={
            "type":      "webpage",
            "url":       url,
            "wordCount": word_count,
            "truncated": truncated,
        },
    )
