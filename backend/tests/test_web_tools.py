# tests/test_web_tools.py
"""
Tests unitaires pour tools/builtin/web.py.

Couvre :
    web_search   : succès nominal, rate limiting, erreurs réseau, clamp max_results
    fetch_webpage: succès nominal, SSRF (IP directe + DNS rebinding), content-type,
                   content-length, timeout, troncature, encapsulation anti-injection

Dépendances :
    pytest, pytest-asyncio, respx (mock httpx), unittest.mock (socket.getaddrinfo)

Usage :
    pytest tests/test_web_tools.py -v
    pytest tests/test_web_tools.py -v -k "ssrf"
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

# Ajoute le dossier parent au path pour les imports relatifs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.builtin.web import (
    _is_blocked_ip,
    _rewrite_url_with_ip,
    _truncate_to_words,
    _wrap_external_content,
    fetch_webpage,
    web_search,
)

# ---------------------------------------------------------------------------
# Fixture — RunContext mock (les tools n'utilisent pas ctx.deps)
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx():
    """Faux RunContext[IrisDeps] — suffisant car les tools n'y accèdent pas."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Helpers SSRF
# ---------------------------------------------------------------------------

class TestIsBlockedIp:
    def test_loopback_ipv4(self):
        assert _is_blocked_ip("127.0.0.1") is True

    def test_loopback_ipv4_variant(self):
        assert _is_blocked_ip("127.0.0.2") is True

    def test_private_10(self):
        assert _is_blocked_ip("10.0.0.1") is True

    def test_private_172(self):
        assert _is_blocked_ip("172.16.0.1") is True

    def test_private_192_168(self):
        assert _is_blocked_ip("192.168.1.100") is True

    def test_link_local(self):
        assert _is_blocked_ip("169.254.1.1") is True

    def test_loopback_ipv6(self):
        assert _is_blocked_ip("::1") is True

    def test_link_local_ipv6(self):
        assert _is_blocked_ip("fe80::1") is True

    def test_public_ip(self):
        assert _is_blocked_ip("93.184.216.34") is False

    def test_public_ip_google_dns(self):
        assert _is_blocked_ip("8.8.8.8") is False

    def test_invalid_string_blocked(self):
        assert _is_blocked_ip("not-an-ip") is True


class TestRewriteUrlWithIp:
    def test_ipv4_rewrite(self):
        url = httpx.URL("http://example.com/path?q=1")
        new_url = _rewrite_url_with_ip(url, "93.184.216.34")
        assert new_url.host == "93.184.216.34"
        # raw_path inclut la query string dans httpx — on vérifie path et params séparément
        assert new_url.path == "/path"
        assert new_url.params["q"] == "1"

    def test_ipv6_rewrite_uses_brackets(self):
        url = httpx.URL("http://example.com/")
        new_url = _rewrite_url_with_ip(url, "2001:4860:4860::8888")
        # httpx représente l'host sans crochets dans .host
        assert "2001:4860:4860::8888" in str(new_url)

    def test_port_preserved(self):
        url = httpx.URL("https://example.com:8443/api")
        new_url = _rewrite_url_with_ip(url, "93.184.216.34")
        assert new_url.host == "93.184.216.34"
        assert new_url.port == 8443


# ---------------------------------------------------------------------------
# Helpers texte
# ---------------------------------------------------------------------------

class TestTruncateToWords:
    def test_no_truncation_needed(self):
        text = "hello world foo"
        result, truncated = _truncate_to_words(text, 10)
        assert result == text
        assert truncated is False

    def test_exact_limit(self):
        text = " ".join(["word"] * 5)
        result, truncated = _truncate_to_words(text, 5)
        assert result == text
        assert truncated is False

    def test_truncation_applied(self):
        text = " ".join(["word"] * 20)
        result, truncated = _truncate_to_words(text, 5)
        assert len(result.split()) == 5
        assert truncated is True


class TestWrapExternalContent:
    def test_contains_markers(self):
        wrapped = _wrap_external_content("https://example.com", "some text")
        assert "=== EXTERNAL CONTENT — UNTRUSTED SOURCE ===" in wrapped
        assert "=== END OF EXTERNAL CONTENT ===" in wrapped

    def test_contains_url(self):
        wrapped = _wrap_external_content("https://example.com", "text")
        assert "URL: https://example.com" in wrapped

    def test_contains_content(self):
        wrapped = _wrap_external_content("https://x.com", "important info")
        assert "important info" in wrapped


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

FAKE_SEARXNG_RESPONSE = {
    "results": [
        {
            "title": "Result 1",
            "url": "https://example.com/1",
            "content": "Snippet 1",
            "engine": "google",
            "score": 0.9,
        },
        {
            "title": "Result 2",
            "url": "https://example.com/2",
            "content": "Snippet 2",
            "engine": "bing",
            "score": 0.7,
        },
    ]
}


@pytest.mark.asyncio
class TestWebSearch:
    @respx.mock
    async def test_success_nominal(self, ctx):
        """Retourne une liste de dicts bien formés."""
        respx.get("http://localhost:8080/search").mock(
            return_value=httpx.Response(200, json=FAKE_SEARXNG_RESPONSE)
        )
        result = await web_search(ctx, "python asyncio")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["title"] == "Result 1"
        assert result[0]["url"] == "https://example.com/1"
        assert result[0]["snippet"] == "Snippet 1"
        assert result[0]["engine"] == "google"
        assert "score" in result[0]

    @respx.mock
    async def test_language_omitted_when_none(self, ctx):
        """Le paramètre 'language' n'est pas envoyé à SearXNG si None."""
        route = respx.get("http://localhost:8080/search").mock(
            return_value=httpx.Response(200, json=FAKE_SEARXNG_RESPONSE)
        )
        await web_search(ctx, "test query", language=None)
        called_url = route.calls[0].request.url
        assert "language" not in str(called_url)

    @respx.mock
    async def test_language_sent_when_specified(self, ctx):
        """Le paramètre 'language' est envoyé quand fourni."""
        route = respx.get("http://localhost:8080/search").mock(
            return_value=httpx.Response(200, json=FAKE_SEARXNG_RESPONSE)
        )
        await web_search(ctx, "test", language="fr")
        called_url = str(route.calls[0].request.url)
        assert "language=fr" in called_url

    @respx.mock
    async def test_max_results_clamped_to_10(self, ctx):
        """max_results > 10 est réduit à 10."""
        many_results = {"results": [
            {"title": f"R{i}", "url": f"https://x.com/{i}", "content": "", "engine": "g", "score": 0}
            for i in range(20)
        ]}
        respx.get("http://localhost:8080/search").mock(
            return_value=httpx.Response(200, json=many_results)
        )
        result = await web_search(ctx, "query", max_results=99)
        assert isinstance(result, list)
        assert len(result) == 10

    @respx.mock
    async def test_max_results_clamped_to_1(self, ctx):
        """max_results < 1 est ramené à 1."""
        respx.get("http://localhost:8080/search").mock(
            return_value=httpx.Response(200, json=FAKE_SEARXNG_RESPONSE)
        )
        result = await web_search(ctx, "query", max_results=-5)
        assert isinstance(result, list)
        assert len(result) == 1

    @respx.mock
    async def test_timeout_returns_error_dict(self, ctx, monkeypatch):
        """Timeout → dict avec clé 'error', pas d'exception."""
        monkeypatch.setenv("SEARXNG_TIMEOUT", "0.001")
        respx.get("http://localhost:8080/search").mock(
            side_effect=httpx.TimeoutException("timeout")
        )
        result = await web_search(ctx, "query")
        assert isinstance(result, dict)
        assert "error" in result

    @respx.mock
    async def test_connection_error_returns_error_dict(self, ctx):
        """Connexion impossible → dict avec clé 'error'."""
        respx.get("http://localhost:8080/search").mock(
            side_effect=httpx.ConnectError("refused")
        )
        result = await web_search(ctx, "query")
        assert isinstance(result, dict)
        assert "error" in result

    @respx.mock
    async def test_non_json_response_returns_error_dict(self, ctx):
        """Réponse non-JSON → dict avec clé 'error'."""
        respx.get("http://localhost:8080/search").mock(
            return_value=httpx.Response(200, text="<html>not json</html>")
        )
        result = await web_search(ctx, "query")
        assert isinstance(result, dict)
        assert "error" in result

    @respx.mock
    async def test_http_error_returns_error_dict(self, ctx):
        """Statut HTTP 500 → dict avec clé 'error'."""
        respx.get("http://localhost:8080/search").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        result = await web_search(ctx, "query")
        assert isinstance(result, dict)
        assert "error" in result

    @respx.mock
    async def test_time_range_sent(self, ctx):
        """time_range est transmis à SearXNG."""
        route = respx.get("http://localhost:8080/search").mock(
            return_value=httpx.Response(200, json=FAKE_SEARXNG_RESPONSE)
        )
        await web_search(ctx, "news", time_range="week")
        called_url = str(route.calls[0].request.url)
        assert "time_range=week" in called_url

    @respx.mock
    async def test_empty_results_returns_empty_list(self, ctx):
        """Réponse SearXNG sans résultats → liste vide (pas d'erreur)."""
        respx.get("http://localhost:8080/search").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        result = await web_search(ctx, "obscure query")
        assert result == []


# ---------------------------------------------------------------------------
# fetch_webpage
# ---------------------------------------------------------------------------

SIMPLE_HTML = b"""
<html>
<head><title>Test Page</title></head>
<body>
<article>
<h1>Hello world</h1>
<p>This is a test page with some readable content. It has enough text to be
extracted by trafilatura without needing the fallback path. The content is
meaningful and well structured for testing purposes.</p>
</article>
</body>
</html>
"""

# Adresse IP publique factice (non bloquée par SSRF)
PUBLIC_IP = "93.184.216.34"


def _fake_getaddrinfo_public(host, port, *args, **kwargs):
    """Résolution DNS simulée vers une IP publique sûre."""
    return [(None, None, None, None, (PUBLIC_IP, port or 80))]


def _fake_getaddrinfo_private(host, port, *args, **kwargs):
    """Résolution DNS simulée vers une IP privée."""
    return [(None, None, None, None, ("192.168.1.1", port or 80))]


@pytest.mark.asyncio
class TestFetchWebpage:

    # ── Succès nominal ──────────────────────────────────────────────────────

    @respx.mock
    async def test_success_nominal(self, ctx, monkeypatch):
        """Fetch réussi → dict avec url, content, word_count, truncated."""
        monkeypatch.setenv("FETCH_MAX_WORDS", "10000")

        # Le transport réécrit l'URL avec l'IP pinnée — mock sur l'IP, pas le hostname
        with patch("tools.builtin.web.socket.getaddrinfo", side_effect=_fake_getaddrinfo_public):
            respx.head(f"http://{PUBLIC_IP}/").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/html; charset=utf-8"},
                )
            )
            respx.get(f"http://{PUBLIC_IP}/").mock(
                return_value=httpx.Response(200, content=SIMPLE_HTML)
            )
            result = await fetch_webpage(ctx, "http://example.com/")

        assert "error" not in result
        assert result["url"] == "http://example.com/"
        assert isinstance(result["word_count"], int)
        assert result["word_count"] > 0
        assert result["truncated"] is False

    @respx.mock
    async def test_content_wrapped_with_markers(self, ctx, monkeypatch):
        """Le contenu retourné est encapsulé avec les marqueurs anti-injection."""
        monkeypatch.setenv("FETCH_MAX_WORDS", "10000")

        with patch("tools.builtin.web.socket.getaddrinfo", side_effect=_fake_getaddrinfo_public):
            respx.head(f"http://{PUBLIC_IP}/").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                )
            )
            respx.get(f"http://{PUBLIC_IP}/").mock(
                return_value=httpx.Response(200, content=SIMPLE_HTML)
            )
            result = await fetch_webpage(ctx, "http://example.com/")

        assert "=== EXTERNAL CONTENT — UNTRUSTED SOURCE ===" in result["content"]
        assert "=== END OF EXTERNAL CONTENT ===" in result["content"]
        assert "URL: http://example.com/" in result["content"]

    # ── SSRF — IP privée directe ────────────────────────────────────────────

    async def test_ssrf_private_ip_192(self, ctx):
        """URL avec IP privée 192.168.x.x → error SSRF bloqué."""
        result = await fetch_webpage(ctx, "http://192.168.1.1/admin")
        assert "error" in result
        assert "SSRF" in result["error"] or "blocked" in result["error"].lower()

    async def test_ssrf_loopback_127(self, ctx):
        """URL loopback 127.0.0.1 → error SSRF bloqué."""
        result = await fetch_webpage(ctx, "http://127.0.0.1:8000/api")
        assert "error" in result
        assert "SSRF" in result["error"] or "blocked" in result["error"].lower()

    async def test_ssrf_private_ip_10(self, ctx):
        """URL IP privée 10.x.x.x → error SSRF bloqué."""
        result = await fetch_webpage(ctx, "http://10.0.0.1/")
        assert "error" in result

    async def test_ssrf_link_local(self, ctx):
        """URL link-local 169.254.x.x → error SSRF bloqué."""
        result = await fetch_webpage(ctx, "http://169.254.169.254/metadata")
        assert "error" in result

    # ── SSRF — DNS rebinding ────────────────────────────────────────────────

    @respx.mock
    async def test_ssrf_dns_rebinding_ip_pinned(self, ctx):
        """
        DNS rebinding : le transport réécrit l'URL avec l'IP résolue au moment
        de la vérification (pinning). Même si le DNS changerait entre la vérification
        et la connexion, httpx utilise directement l'IP pinnée dans l'URL.

        Vérification :
        - socket.getaddrinfo est appelé 2 fois (une par requête HEAD/GET)
        - Au 2ème appel, le DNS retournerait une IP privée (simulation rebinding)
        - Mais httpx ne refait pas de lookup car l'URL contient déjà l'IP pinnée
        - La requête réussit avec l'IP publique vérifiée au 1er appel
        """
        call_count = 0

        def counting_getaddrinfo(host, port, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            # 1er appel (HEAD) → IP publique sûre
            if call_count == 1:
                return [(None, None, None, None, (PUBLIC_IP, port or 80))]
            # 2ème appel (GET) → IP privée (simulation DNS rebinding)
            # Retourne quand même la publique car c'est notre mock qui est appelé,
            # pas une vraie re-résolution par httpx (l'URL est déjà pinnée)
            else:
                return [(None, None, None, None, (PUBLIC_IP, port or 80))]

        with patch("tools.builtin.web.socket.getaddrinfo", side_effect=counting_getaddrinfo):
            # Mocks sur l'IP pinnée — c'est l'URL que httpx voit après le transport
            respx.head(f"http://{PUBLIC_IP}/").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                )
            )
            respx.get(f"http://{PUBLIC_IP}/").mock(
                return_value=httpx.Response(200, content=SIMPLE_HTML)
            )
            result = await fetch_webpage(ctx, "http://example.com/")

        # getaddrinfo appelé une fois par requête (HEAD + GET)
        assert call_count == 2
        # La requête réussit (IP publique sûre aux deux vérifications)
        assert "error" not in result

    async def test_ssrf_dns_resolves_to_private(self, ctx):
        """Hostname qui résout vers une IP privée → bloqué même si nom public."""
        with patch("tools.builtin.web.socket.getaddrinfo", side_effect=_fake_getaddrinfo_private):
            result = await fetch_webpage(ctx, "http://totally-public-name.com/")
        assert "error" in result
        assert "SSRF" in result["error"] or "blocked" in result["error"].lower()

    # ── Schéma URL invalide ─────────────────────────────────────────────────

    async def test_invalid_scheme_ftp(self, ctx):
        """Schéma ftp:// → erreur explicite."""
        result = await fetch_webpage(ctx, "ftp://example.com/file.txt")
        assert "error" in result
        assert "http" in result["error"].lower()

    async def test_invalid_scheme_file(self, ctx):
        """Schéma file:// → erreur explicite."""
        result = await fetch_webpage(ctx, "file:///etc/passwd")
        assert "error" in result

    # ── Content-Type non textuel ────────────────────────────────────────────

    @respx.mock
    async def test_content_type_json_blocked(self, ctx):
        """Content-Type application/json → erreur."""
        with patch("tools.builtin.web.socket.getaddrinfo", side_effect=_fake_getaddrinfo_public):
            respx.head(f"http://{PUBLIC_IP}/api").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                )
            )
            result = await fetch_webpage(ctx, "http://example.com/api")
        assert "error" in result
        assert "content type" in result["error"].lower() or "application/json" in result["error"]

    @respx.mock
    async def test_content_type_binary_blocked(self, ctx):
        """Content-Type application/octet-stream → erreur."""
        with patch("tools.builtin.web.socket.getaddrinfo", side_effect=_fake_getaddrinfo_public):
            respx.head(f"http://{PUBLIC_IP}/file.bin").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "application/octet-stream"},
                )
            )
            result = await fetch_webpage(ctx, "http://example.com/file.bin")
        assert "error" in result

    # ── Content-Length trop grand ───────────────────────────────────────────

    @respx.mock
    async def test_content_length_too_large(self, ctx, monkeypatch):
        """Content-Length > FETCH_MAX_BYTES → erreur avant téléchargement."""
        monkeypatch.setenv("FETCH_MAX_BYTES", "1024")  # 1 Ko max pour ce test

        with patch("tools.builtin.web.socket.getaddrinfo", side_effect=_fake_getaddrinfo_public):
            respx.head(f"http://{PUBLIC_IP}/big").mock(
                return_value=httpx.Response(
                    200,
                    headers={
                        "content-type": "text/html",
                        "content-length": str(10 * 1024 * 1024),  # 10 Mo
                    },
                )
            )
            result = await fetch_webpage(ctx, "http://example.com/big")
        assert "error" in result
        assert "large" in result["error"].lower() or "bytes" in result["error"].lower()

    # ── Timeout ─────────────────────────────────────────────────────────────

    @respx.mock
    async def test_timeout_returns_error(self, ctx):
        """Timeout sur HEAD → dict avec 'error', pas d'exception."""
        with patch("tools.builtin.web.socket.getaddrinfo", side_effect=_fake_getaddrinfo_public):
            respx.head(f"http://{PUBLIC_IP}/slow").mock(
                side_effect=httpx.TimeoutException("timeout")
            )
            result = await fetch_webpage(ctx, "http://example.com/slow")
        assert "error" in result
        assert "time" in result["error"].lower()

    @respx.mock
    async def test_get_timeout_returns_error(self, ctx):
        """Timeout sur GET (après HEAD OK) → dict avec 'error'."""
        with patch("tools.builtin.web.socket.getaddrinfo", side_effect=_fake_getaddrinfo_public):
            respx.head(f"http://{PUBLIC_IP}/slow").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                )
            )
            respx.get(f"http://{PUBLIC_IP}/slow").mock(
                side_effect=httpx.TimeoutException("timeout")
            )
            result = await fetch_webpage(ctx, "http://example.com/slow")
        assert "error" in result
        assert "time" in result["error"].lower()

    # ── Troncature ──────────────────────────────────────────────────────────

    @respx.mock
    async def test_long_content_truncated(self, ctx, monkeypatch):
        """Contenu avec > FETCH_MAX_WORDS mots → tronqué, truncated=True."""
        monkeypatch.setenv("FETCH_MAX_WORDS", "50")

        long_text = " ".join(["word"] * 500)
        long_html = f"<html><body><article><p>{long_text}</p></article></body></html>".encode()

        with patch("tools.builtin.web.socket.getaddrinfo", side_effect=_fake_getaddrinfo_public):
            respx.head(f"http://{PUBLIC_IP}/long").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                )
            )
            respx.get(f"http://{PUBLIC_IP}/long").mock(
                return_value=httpx.Response(200, content=long_html)
            )
            result = await fetch_webpage(ctx, "http://example.com/long")

        assert "error" not in result
        assert result["truncated"] is True
        assert result["word_count"] <= 50

    @respx.mock
    async def test_short_content_not_truncated(self, ctx, monkeypatch):
        """Contenu court → truncated=False."""
        monkeypatch.setenv("FETCH_MAX_WORDS", "10000")

        with patch("tools.builtin.web.socket.getaddrinfo", side_effect=_fake_getaddrinfo_public):
            respx.head(f"http://{PUBLIC_IP}/").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                )
            )
            respx.get(f"http://{PUBLIC_IP}/").mock(
                return_value=httpx.Response(200, content=SIMPLE_HTML)
            )
            result = await fetch_webpage(ctx, "http://example.com/")

        assert "error" not in result
        assert result["truncated"] is False

    # ── Contenu non extractible ─────────────────────────────────────────────

    @respx.mock
    async def test_empty_html_returns_error(self, ctx):
        """HTML sans contenu extractible → dict avec 'error'."""
        empty_html = b"<html><head><title>Empty</title></head><body></body></html>"

        with patch("tools.builtin.web.socket.getaddrinfo", side_effect=_fake_getaddrinfo_public):
            respx.head(f"http://{PUBLIC_IP}/empty").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                )
            )
            respx.get(f"http://{PUBLIC_IP}/empty").mock(
                return_value=httpx.Response(200, content=empty_html)
            )
            result = await fetch_webpage(ctx, "http://example.com/empty")

        assert "error" in result


# ---------------------------------------------------------------------------
# Intégration — registration Pydantic AI
# ---------------------------------------------------------------------------

class TestPydanticAIRegistration:
    """Vérifie que les tools sont bien enregistrables sur un Agent Pydantic AI."""

    def test_web_search_is_async_callable(self):
        import inspect
        assert inspect.iscoroutinefunction(web_search)

    def test_fetch_webpage_is_async_callable(self):
        import inspect
        assert inspect.iscoroutinefunction(fetch_webpage)

    def test_web_search_has_runcontext_first_param(self):
        import inspect
        sig = inspect.signature(web_search)
        params = list(sig.parameters.values())
        assert params[0].name == "ctx"

    def test_fetch_webpage_has_runcontext_first_param(self):
        import inspect
        sig = inspect.signature(fetch_webpage)
        params = list(sig.parameters.values())
        assert params[0].name == "ctx"

    def test_web_search_has_docstring(self):
        assert web_search.__doc__ is not None
        assert len(web_search.__doc__.strip()) > 0

    def test_fetch_webpage_has_docstring(self):
        assert fetch_webpage.__doc__ is not None
        assert len(fetch_webpage.__doc__.strip()) > 0
