"""
Tests Phase 3.1 — Tool Calling Foundation (sans modèle LLM requis).

Ces tests couvrent toutes les couches du stack tool calling :
    3.1.1 — ToolSchema / ToolParameter (Pydantic v2, format OpenAI)
    3.1.2 — ToolRegistry (thread-safety, persistence, protection système)
    3.1.3 — Outils builtin (get_current_time, get_system_info, ping_host)
    3.1.4 — MinistralEngine.stream_with_tools() (parsing, boucle, events)

Les tests E2E avec le vrai modèle (3.1.5) sont déclenchés manuellement via
la variable d'environnement MODEL_PATH :
    MODEL_PATH=/chemin/vers/modele pytest tests/test_tool_calling_e2e.py -v -k e2e

Usage :
    python tests/test_tool_calling_e2e.py         # tous les tests unitaires
    pytest tests/test_tool_calling_e2e.py -v      # format pytest verbose
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import MinistralEngine
from core.tools import (
    PermissionLevel,
    SystemToolProtectedError,
    ToolParameter,
    ToolRegistry,
    ToolSchema,
)
from tools.builtin.system import (
    BUILTIN_TOOLS,
    get_current_time,
    get_system_info,
    ping_host,
)


# ── Helpers fixtures ──────────────────────────────────────────────────────────


def _make_isolated_registry(tmp_path: Path) -> ToolRegistry:
    """Crée une instance ToolRegistry isolée (ne touche pas le singleton)."""
    return ToolRegistry(storage_path=tmp_path / "tools")


def _make_engine_mock() -> MinistralEngine:
    """MinistralEngine sans modèle chargé, prêt pour monkey-patch de _generate_raw."""
    engine = MinistralEngine.__new__(MinistralEngine)
    engine._is_loaded = True
    engine.model_path = "mock"
    return engine


def run(coro):
    """Raccourci asyncio.run() pour tests synchrones."""
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3.1.1 — ToolSchema / ToolParameter
# ═══════════════════════════════════════════════════════════════════════════════


def test_schema_openai_format_exact():
    """to_openai_format() génère la structure exacte attendue par OpenAI/Mistral."""
    schema = ToolSchema(
        name="get_time",
        description="Get current time",
        parameters=[
            ToolParameter(
                name="timezone",
                type="string",
                description="IANA timezone",
                required=True,
            )
        ],
    )
    out = schema.to_openai_format()

    assert out["type"] == "function"
    assert "function" in out
    assert out["function"]["name"] == "get_time"
    assert out["function"]["description"] == "Get current time"
    params = out["function"]["parameters"]
    assert params["type"] == "object"
    assert "timezone" in params["properties"]
    assert params["properties"]["timezone"]["type"] == "string"
    assert "timezone" in params["required"]
    print("[OK] to_openai_format() structure exacte validée")


def test_schema_name_snake_case():
    """Nom CamelCase ou avec tirets refusé."""
    import pytest

    try:
        ToolSchema(name="GetTime", description="...")
        assert False, "Doit lever ValidationError"
    except Exception as e:
        assert "pattern" in str(e).lower() or "validation" in str(e).lower()

    try:
        ToolSchema(name="get-time", description="...")
        assert False, "Doit lever ValidationError"
    except Exception:
        pass

    # Valide
    ToolSchema(name="get_time_v2", description="...")
    ToolSchema(name="get_current_time", description="...")
    print("[OK] validation snake_case OK")


def test_schema_json_types_strict():
    """Seuls les 5 types JSON Schema valides sont acceptés."""
    for valid_type in ("string", "number", "boolean", "array", "object"):
        p = ToolParameter(name="x", type=valid_type)
        assert p.type == valid_type

    for invalid_type in ("datetime", "int", "float", "str", "list", "dict", "any"):
        try:
            ToolParameter(name="x", type=invalid_type)
            assert False, f"Type '{invalid_type}' aurait dû être rejeté"
        except Exception:
            pass
    print("[OK] types JSON Schema stricts validés")


def test_schema_datetime_round_trip():
    """created_at serialisé en ISO 8601 et désérialisé sans perte timezone/microsecond."""
    now = datetime.now(timezone.utc)
    schema = ToolSchema(name="test_tool", description="...", created_at=now)

    json_str = schema.model_dump_json()
    schema2 = ToolSchema.model_validate_json(json_str)

    assert schema.created_at == schema2.created_at, (
        f"Round-trip échoué : {schema.created_at} != {schema2.created_at}"
    )
    # Vérifier que le microsecond est préservé
    assert schema.created_at.microsecond == schema2.created_at.microsecond
    print("[OK] datetime round-trip (timezone + microseconde) OK")


def test_schema_naive_datetime_rejected():
    """datetime naïf (sans timezone) refusé par AwareDatetime."""
    try:
        ToolSchema(name="test", description="...", created_at=datetime.now())
        assert False, "datetime naïf doit être rejeté"
    except Exception:
        pass
    print("[OK] datetime naïf rejeté")


def test_schema_enum_in_output():
    """Paramètre avec enum correctement sérialisé dans to_openai_format()."""
    param = ToolParameter(
        name="timezone",
        type="string",
        enum=["Europe/Paris", "America/New_York", "Asia/Tokyo"],
        required=True,
    )
    schema = ToolSchema(name="get_time", description="...", parameters=[param])
    out = schema.to_openai_format()

    props = out["function"]["parameters"]["properties"]
    assert "enum" in props["timezone"]
    assert props["timezone"]["enum"] == ["Europe/Paris", "America/New_York", "Asia/Tokyo"]
    print("[OK] enum sérialisé dans output OK")


def test_schema_duplicate_param_names():
    """Deux paramètres avec le même nom refusés."""
    try:
        ToolSchema(
            name="test",
            description="...",
            parameters=[
                ToolParameter(name="x", type="string"),
                ToolParameter(name="x", type="number"),
            ],
        )
        assert False, "Doublons auraient dû être refusés"
    except Exception:
        pass
    print("[OK] noms dupliqués refusés")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3.1.2 — ToolRegistry
# ═══════════════════════════════════════════════════════════════════════════════


def test_registry_singleton():
    """get_instance() retourne toujours la même instance."""
    ToolRegistry._reset_instance()
    with tempfile.TemporaryDirectory() as tmp:
        r1 = ToolRegistry.get_instance(Path(tmp) / "t")
        r2 = ToolRegistry.get_instance()
        assert r1 is r2
    ToolRegistry._reset_instance()
    print("[OK] singleton OK")


def test_registry_register_persist():
    """register() crée l'outil et le persist en JSON."""
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            reg = _make_isolated_registry(Path(tmp))
            schema = ToolSchema(name="test_reg", description="Test")
            await reg.register(schema, lambda: "result")

            data = json.loads((Path(tmp) / "tools" / "registry.json").read_text())
            assert "test_reg" in data["tools"]
            assert data["version"] == 1
    run(_run())
    print("[OK] register + persist JSON OK")


def test_registry_force_flag():
    """register sans force refuse doublons ; avec force=True autorise."""
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            reg = _make_isolated_registry(Path(tmp))
            schema = ToolSchema(name="dup_tool", description="...")
            await reg.register(schema, lambda: "v1")

            try:
                await reg.register(schema, lambda: "v2")
                assert False, "Doit lever ValueError"
            except ValueError:
                pass

            await reg.register(schema, lambda: "v2", force=True)
            assert reg.has_executor("dup_tool")
    run(_run())
    print("[OK] force flag OK")


def test_registry_system_tool_protection():
    """Outil created_by='system' ne peut être ni écrasé ni supprimé."""
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            reg = _make_isolated_registry(Path(tmp))
            sys_schema = ToolSchema(
                name="sys_tool",
                description="...",
                created_by="system",
            )
            await reg.register(sys_schema, lambda: None)

            # unregister refusé (retourne False)
            result = await reg.unregister("sys_tool")
            assert result is False

            # register avec force=True refusé (SystemToolProtectedError)
            try:
                await reg.register(sys_schema, lambda: "v2", force=True)
                assert False, "Doit lever SystemToolProtectedError"
            except SystemToolProtectedError:
                pass

            # L'outil est toujours là
            assert reg.get("sys_tool") is not None
    run(_run())
    print("[OK] protection system tools OK")


def test_registry_openai_format_strict():
    """get_tools_for_ministral() retourne uniquement format OpenAI pur."""
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            reg = _make_isolated_registry(Path(tmp))
            schema = ToolSchema(
                name="my_tool",
                description="...",
                parameters=[ToolParameter(name="x", type="string", required=True)],
                permission_level=PermissionLevel.WRITE_SAFE,
                created_by="user",
            )
            await reg.register(schema, lambda x: x)
            tools = reg.get_tools_for_ministral()

            assert len(tools) == 1
            t = tools[0]
            assert t["type"] == "function"
            assert "function" in t
            assert t["function"]["parameters"]["type"] == "object"
            # Métadonnées étendues absentes
            assert "permission_level" not in t
            assert "created_by" not in t
            assert "success_count" not in t
    run(_run())
    print("[OK] format OpenAI pur (métadonnées absentes) OK")


def test_registry_permission_filter():
    """get_tools_for_ministral et get_tools_up_to_level filtrent correctement."""
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            reg = _make_isolated_registry(Path(tmp))
            await reg.register(
                ToolSchema(name="r_only", description="...", permission_level=PermissionLevel.READ_ONLY),
                lambda: None,
            )
            await reg.register(
                ToolSchema(name="w_safe", description="...", permission_level=PermissionLevel.WRITE_SAFE),
                lambda: None,
            )
            await reg.register(
                ToolSchema(name="network", description="...", permission_level=PermissionLevel.NETWORK),
                lambda: None,
            )

            # Filtre exact
            assert len(reg.get_tools_for_ministral(PermissionLevel.READ_ONLY)) == 1
            assert len(reg.get_tools_for_ministral()) == 3

            # up_to_level
            up_to_write = reg.get_tools_up_to_level(PermissionLevel.WRITE_SAFE)
            assert len(up_to_write) == 2
            names = {t["function"]["name"] for t in up_to_write}
            assert "r_only" in names and "w_safe" in names
    run(_run())
    print("[OK] filtrage permission OK")


def test_registry_persistence_reload():
    """Schemas survivent au redémarrage ; executors doivent être re-registered."""
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "tools"
            reg = ToolRegistry(storage_path=storage)
            p = ToolParameter(name="city", type="string", required=True)
            schema = ToolSchema(name="weather_tool", description="Weather", parameters=[p])
            await reg.register(schema, lambda city: f"sunny in {city}")

            # Reload
            reg2 = ToolRegistry(storage_path=storage)
            reloaded = reg2.get("weather_tool")
            assert reloaded is not None
            assert reloaded.description == "Weather"
            assert not reg2.has_executor("weather_tool")  # executor absent après reload
    run(_run())
    print("[OK] persistence reload OK")


def test_registry_concurrent_safety():
    """100 opérations concurrentes sans corruption."""
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            reg = _make_isolated_registry(Path(tmp))
            schemas = [ToolSchema(name=f"tool_{i}", description="...") for i in range(100)]
            await asyncio.gather(*[reg.register(s, lambda: None) for s in schemas])
            assert reg.tool_count == 100, f"Attendu 100, obtenu {reg.tool_count}"
    run(_run())
    print("[OK] 100 concurrent ops sans corruption OK")


def test_registry_update_stats():
    """update_stats incrémente correctement succès/échecs et moyenne."""
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            reg = _make_isolated_registry(Path(tmp))
            await reg.register(ToolSchema(name="stats_t", description="..."), lambda: None)

            await reg.update_stats("stats_t", success=True, duration_ms=10.0)
            await reg.update_stats("stats_t", success=True, duration_ms=20.0)
            await reg.update_stats("stats_t", success=False, duration_ms=5.0)

            s = reg.get("stats_t")
            assert s.success_count == 2
            assert s.failure_count == 1
            assert s.last_used is not None
            assert abs(s.average_duration_ms - 11.67) < 0.1
    run(_run())
    print("[OK] update_stats OK")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3.1.3 — Outils Builtin
# ═══════════════════════════════════════════════════════════════════════════════


def test_builtin_get_current_time_valid():
    """get_current_time retourne ISO 8601 avec timezone."""
    async def _run():
        result = await get_current_time("Europe/Paris")
        assert isinstance(result, str)
        assert "T" in result
        assert ("+01:00" in result or "+02:00" in result), result

        result_utc = await get_current_time("UTC")
        assert "+00:00" in result_utc

        result_default = await get_current_time()
        assert "+00:00" in result_default
    run(_run())
    print("[OK] get_current_time valide OK")


def test_builtin_get_current_time_invalid_tz():
    """get_current_time timezone invalide → erreur gracieuse, pas de crash."""
    async def _run():
        result = await get_current_time("Invalid/Timezone")
        assert isinstance(result, str)
        assert "error" in result.lower() or "Error" in result
    run(_run())
    print("[OK] get_current_time timezone invalide OK")


def test_builtin_get_system_info():
    """get_system_info retourne métriques attendues sans données sensibles."""
    async def _run():
        result = await get_system_info()
        assert isinstance(result, dict)
        assert "cpu_percent" in result
        assert "memory_used_gb" in result
        assert "memory_total_gb" in result
        assert "memory_percent" in result
        assert "disk_used_gb" in result
        assert "disk_total_gb" in result
        assert "cpu_count" in result

        # Sécurité : données sensibles filtrées
        result_str = str(result).lower()
        assert "hostname" not in result_str, f"hostname exposé: {result}"
        assert "username" not in result_str, f"username exposé: {result}"
        # Note: on ne vérifie pas 192.168 car les dicts ne contiennent pas d'IP
    run(_run())
    print("[OK] get_system_info (métriques + filtrage sensibles) OK")


def test_builtin_ping_valid():
    """ping_host hôte valide retourne success=True + latency."""
    async def _run():
        result = await ping_host("8.8.8.8", timeout=5.0)
        assert isinstance(result, dict)
        assert result.get("success") is True, result
        assert result.get("latency_ms", 0) > 0
    run(_run())
    print(f"[OK] ping_host 8.8.8.8 → success OK")


def test_builtin_ping_invalid():
    """ping_host hôte invalide retourne success=False + error, pas de crash."""
    async def _run():
        result = await ping_host("definitely.invalid.host.xyz123", timeout=2.0)
        assert isinstance(result, dict)
        assert result.get("success") is False, result
        assert "error" in result, result
    run(_run())
    print("[OK] ping_host hôte invalide OK")


def test_builtin_auto_registration():
    """register_builtin_tools() enregistre 3 outils au startup."""
    async def _run():
        from tools.builtin import register_builtin_tools

        with tempfile.TemporaryDirectory() as tmp:
            ToolRegistry._reset_instance()
            reg = ToolRegistry(storage_path=Path(tmp) / "tools")
            await register_builtin_tools(reg)

            tools = reg.get_tools_for_ministral()
            names = {t["function"]["name"] for t in tools}

            assert "get_current_time" in names
            assert "get_system_info" in names
            assert "ping_host" in names

            # Idempotent
            await register_builtin_tools(reg)
            assert reg.tool_count == 3

            # Tous created_by=system, READ_ONLY
            for name in names:
                s = reg.get(name)
                assert s.created_by == "system"
                assert s.permission_level == PermissionLevel.READ_ONLY

            ToolRegistry._reset_instance()
    run(_run())
    print("[OK] auto-registration 3 outils builtin OK")


def test_builtin_openai_schema_valid():
    """Schémas des outils builtin conformes au format OpenAI strict."""
    for name, (schema, _) in BUILTIN_TOOLS.items():
        out = schema.to_openai_format()
        assert out["type"] == "function", f"{name}: type incorrect"
        assert out["function"]["parameters"]["type"] == "object", f"{name}: parameters.type incorrect"
        assert isinstance(out["function"]["parameters"]["properties"], dict)
        assert isinstance(out["function"]["parameters"]["required"], list)
    print("[OK] schémas builtin conformes OpenAI")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3.1.4 — MinistralEngine.stream_with_tools
# ═══════════════════════════════════════════════════════════════════════════════


def _build_engine_with_registry(tmp_path: Path):
    """Retourne (engine_mock, registry) avec les 3 outils builtin enregistrés."""
    async def _setup():
        reg = _make_isolated_registry(tmp_path)
        from tools.builtin import register_builtin_tools
        await register_builtin_tools(reg)
        return reg

    reg = asyncio.run(_setup())
    engine = _make_engine_mock()
    return engine, reg


def test_engine_text_response_no_tools():
    """Réponse texte pure → events text_chunk + final, pas de tool_call."""
    with tempfile.TemporaryDirectory() as tmp:
        engine, reg = _build_engine_with_registry(Path(tmp))
        tools = reg.get_tools_for_ministral()

        engine._generate_raw = lambda *a, **k: "La réponse est 42."

        async def _run():
            events = [e async for e in engine.stream_with_tools(
                [{"role": "user", "content": "Qu'est-ce que 6×7 ?"}],
                tools, reg,
            )]
            types = {e["type"] for e in events}
            assert "final" in types
            assert "tool_call" not in types
            assert "tool_result" not in types
        asyncio.run(_run())
    print("[OK] texte pur → pas de tool_call OK")


def test_engine_single_tool_call():
    """Tool call simple → séquence tool_call → tool_result → final."""
    with tempfile.TemporaryDirectory() as tmp:
        engine, reg = _build_engine_with_registry(Path(tmp))
        tools = reg.get_tools_for_ministral()

        call_count = [0]

        def mock_gen(*a, **k):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"tool_calls": [{"name": "get_current_time", "arguments": {"timezone": "UTC"}}]})
            return "Il est 10h00 UTC."

        engine._generate_raw = mock_gen

        async def _run():
            events = [e async for e in engine.stream_with_tools(
                [{"role": "user", "content": "Quelle heure est-il ?"}],
                tools, reg,
            )]
            tool_calls = [e for e in events if e["type"] == "tool_call"]
            tool_results = [e for e in events if e["type"] == "tool_result"]
            finals = [e for e in events if e["type"] == "final"]

            assert len(tool_calls) == 1, tool_calls
            assert tool_calls[0]["name"] == "get_current_time"
            assert len(tool_results) == 1, tool_results
            # Résultat contient une heure ISO 8601
            assert "T" in str(tool_results[0]["result"]) or "error" in str(tool_results[0]["result"]).lower()
            assert len(finals) == 1
        asyncio.run(_run())
    print("[OK] simple tool call (tool_call + tool_result + final) OK")


def test_engine_multi_tool_calls():
    """Deux tool calls dans une seule réponse → deux tool_result."""
    with tempfile.TemporaryDirectory() as tmp:
        engine, reg = _build_engine_with_registry(Path(tmp))
        tools = reg.get_tools_for_ministral()

        call_count = [0]

        def mock_gen(*a, **k):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"tool_calls": [
                    {"name": "get_current_time", "arguments": {"timezone": "Europe/Paris"}},
                    {"name": "get_current_time", "arguments": {"timezone": "Asia/Tokyo"}},
                ]})
            return "Il est X à Paris et Y à Tokyo."

        engine._generate_raw = mock_gen

        async def _run():
            events = [e async for e in engine.stream_with_tools(
                [{"role": "user", "content": "Heure à Paris et Tokyo ?"}],
                tools, reg,
            )]
            tool_calls = [e for e in events if e["type"] == "tool_call"]
            tool_results = [e for e in events if e["type"] == "tool_result"]

            assert len(tool_calls) == 2, f"Attendu 2 tool_calls, obtenu {tool_calls}"
            assert len(tool_results) == 2
            timezones = {tc["arguments"].get("timezone") for tc in tool_calls}
            assert "Europe/Paris" in timezones
            assert "Asia/Tokyo" in timezones
        asyncio.run(_run())
    print("[OK] multi-tool calls (2 tool_result) OK")


def test_engine_parsing_mixed_text():
    """JSON mixte avec texte avant/après correctement parsé."""
    with tempfile.TemporaryDirectory() as tmp:
        engine, reg = _build_engine_with_registry(Path(tmp))
        tools = reg.get_tools_for_ministral()

        call_count = [0]

        def mock_gen(*a, **k):
            call_count[0] += 1
            if call_count[0] == 1:
                return 'Laisse-moi vérifier... {"tool_calls": [{"name": "get_system_info", "arguments": {}}]} Voilà !'
            return "Votre CPU tourne à 23%."

        engine._generate_raw = mock_gen

        async def _run():
            events = [e async for e in engine.stream_with_tools(
                [{"role": "user", "content": "État du système ?"}],
                tools, reg,
            )]
            tool_calls = [e for e in events if e["type"] == "tool_call"]
            assert len(tool_calls) == 1, f"Parser n'a pas extrait le tool call du texte mixte: {events}"
            assert tool_calls[0]["name"] == "get_system_info"
        asyncio.run(_run())
    print("[OK] parsing JSON mixte avec texte OK")


def test_engine_tool_calls_token_format():
    """Format [TOOL_CALLS] Mistral natif correctement parsé."""
    with tempfile.TemporaryDirectory() as tmp:
        engine, reg = _build_engine_with_registry(Path(tmp))
        tools = reg.get_tools_for_ministral()

        call_count = [0]

        def mock_gen(*a, **k):
            call_count[0] += 1
            if call_count[0] == 1:
                return '[TOOL_CALLS] [{"name": "get_current_time", "arguments": {"timezone": "UTC"}}]'
            return "Heure obtenue."

        engine._generate_raw = mock_gen

        async def _run():
            events = [e async for e in engine.stream_with_tools(
                [{"role": "user", "content": "test"}],
                tools, reg,
            )]
            tool_calls = [e for e in events if e["type"] == "tool_call"]
            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "get_current_time"
        asyncio.run(_run())
    print("[OK] format [TOOL_CALLS] Mistral natif OK")


def test_engine_unknown_tool_graceful():
    """Outil hallucinated → error injecté dans contexte, pas de crash."""
    with tempfile.TemporaryDirectory() as tmp:
        engine, reg = _build_engine_with_registry(Path(tmp))
        tools = reg.get_tools_for_ministral()

        call_count = [0]

        def mock_gen(*a, **k):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]})
            return "Je ne peux pas récupérer la météo."

        engine._generate_raw = mock_gen

        async def _run():
            events = [e async for e in engine.stream_with_tools(
                [{"role": "user", "content": "Météo à Paris ?"}],
                tools, reg,
            )]
            tool_results = [e for e in events if e["type"] == "tool_result"]
            assert len(tool_results) >= 1
            result = tool_results[0]["result"]
            assert isinstance(result, dict)
            assert result.get("success") is False
            assert "get_weather" in result.get("error", "")
            assert "available_tools" in result
        asyncio.run(_run())
    print("[OK] outil inexistant → error gracieux OK")


def test_engine_max_iterations():
    """Hard limit max_iterations respectée, event error en sortie."""
    with tempfile.TemporaryDirectory() as tmp:
        engine, reg = _build_engine_with_registry(Path(tmp))
        tools = reg.get_tools_for_ministral()

        # Boucle infinie simulée : toujours retourner un tool call
        engine._generate_raw = lambda *a, **k: json.dumps(
            {"tool_calls": [{"name": "get_current_time", "arguments": {"timezone": "UTC"}}]}
        )

        async def _run():
            events = [e async for e in engine.stream_with_tools(
                [{"role": "user", "content": "test"}],
                tools, reg, max_iterations=3,
            )]
            error_events = [e for e in events if e["type"] == "error"]
            assert len(error_events) >= 1, events
            msg = error_events[-1]["content"].lower()
            assert "iteration" in msg or "limite" in msg
            # Exactly 3 tool_call cycles
            tool_calls = [e for e in events if e["type"] == "tool_call"]
            assert len(tool_calls) == 3, f"Attendu 3 iterations, obtenu {len(tool_calls)}"
        asyncio.run(_run())
    print("[OK] max_iterations=3 stoppé OK")


def test_engine_all_event_types():
    """Tous les types d'events attendus sont émis dans un flux complet."""
    with tempfile.TemporaryDirectory() as tmp:
        engine, reg = _build_engine_with_registry(Path(tmp))
        tools = reg.get_tools_for_ministral()

        call_count = [0]

        def mock_gen(*a, **k):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"tool_calls": [{"name": "get_system_info", "arguments": {}}]})
            return "CPU à 23%."

        engine._generate_raw = mock_gen

        async def _run():
            events = [e async for e in engine.stream_with_tools(
                [{"role": "user", "content": "État ?"}], tools, reg,
            )]
            types = {e["type"] for e in events}
            assert "tool_call" in types, types
            assert "tool_result" in types, types
            assert "final" in types, types
            assert "text_chunk" in types or "final" in types, types
        asyncio.run(_run())
    print("[OK] tous types events présents OK")


def test_engine_backward_compat():
    """stream(), generate(), stream_messages() inchangés et fonctionnels."""
    import inspect
    engine = MinistralEngine("/tmp/mock")
    assert not inspect.iscoroutinefunction(engine.stream)
    assert not inspect.iscoroutinefunction(engine.generate)
    assert not inspect.iscoroutinefunction(engine.stream_messages)
    assert inspect.iscoroutinefunction(engine.load)
    assert inspect.isasyncgenfunction(engine.stream_with_tools)
    print("[OK] backward compat signatures OK")


def test_engine_model_not_loaded():
    """stream_with_tools avec modèle non chargé → event error, pas de crash."""
    async def _run():
        engine = MinistralEngine("/tmp/fake")
        # _is_loaded = False par défaut

        with tempfile.TemporaryDirectory() as tmp:
            reg = _make_isolated_registry(Path(tmp))
            events = [e async for e in engine.stream_with_tools(
                [{"role": "user", "content": "test"}], [], reg,
            )]
            assert events[0]["type"] == "error"
    asyncio.run(_run())
    print("[OK] modèle non chargé → error gracieux OK")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3.1.5 — Tests E2E avec vrai modèle (MODEL_PATH requis)
# ═══════════════════════════════════════════════════════════════════════════════


def _skip_if_no_model():
    """Retourne True si MODEL_PATH non défini (skip E2E)."""
    return not os.getenv("MODEL_PATH", "").strip()


def test_e2e_simple_tool_call():
    """E2E: 'Quelle heure est-il ?' → get_current_time appelé."""
    if _skip_if_no_model():
        print("[SKIP] test_e2e_simple_tool_call (MODEL_PATH non défini)")
        return

    model_path = os.environ["MODEL_PATH"]

    async def _run():
        from tools.builtin import register_builtin_tools

        ToolRegistry._reset_instance()
        with tempfile.TemporaryDirectory() as tmp:
            reg = ToolRegistry(storage_path=Path(tmp) / "tools")
            await register_builtin_tools(reg)
            tools = reg.get_tools_for_ministral()

            engine = MinistralEngine(model_path)
            await engine.load()

            messages = [{"role": "user", "content": "Quelle heure est-il exactement en UTC ?"}]

            events = []
            t0 = time.perf_counter()
            async for event in engine.stream_with_tools(messages, tools, reg, temperature=0.15):
                events.append(event)
            elapsed = time.perf_counter() - t0

            types = {e["type"] for e in events}
            tool_calls = [e for e in events if e["type"] == "tool_call"]

            print(f"  Events: {[e['type'] for e in events]}")
            print(f"  Elapsed: {elapsed:.1f}s")

            assert "tool_call" in types, f"Modèle n'a pas appelé d'outil. Events: {events}"
            assert any(tc["name"] == "get_current_time" for tc in tool_calls), tool_calls
            assert "tool_result" in types
            assert "final" in types

            ToolRegistry._reset_instance()
    asyncio.run(_run())
    print("[OK] E2E simple tool call OK")


def test_e2e_system_info():
    """E2E: 'Quel % CPU ?' → get_system_info appelé."""
    if _skip_if_no_model():
        print("[SKIP] test_e2e_system_info (MODEL_PATH non défini)")
        return

    model_path = os.environ["MODEL_PATH"]

    async def _run():
        from tools.builtin import register_builtin_tools

        ToolRegistry._reset_instance()
        with tempfile.TemporaryDirectory() as tmp:
            reg = ToolRegistry(storage_path=Path(tmp) / "tools")
            await register_builtin_tools(reg)
            tools = reg.get_tools_for_ministral()

            engine = MinistralEngine(model_path)
            await engine.load()

            messages = [{"role": "user", "content": "Quel est l'utilisation CPU actuelle de ma machine ?"}]

            events = []
            async for event in engine.stream_with_tools(messages, tools, reg, temperature=0.15):
                events.append(event)

            types = {e["type"] for e in events}
            tool_calls = [e for e in events if e["type"] == "tool_call"]

            print(f"  Events: {[e['type'] for e in events]}")
            assert "tool_call" in types, f"Events: {events}"
            assert any(tc["name"] == "get_system_info" for tc in tool_calls), tool_calls

            ToolRegistry._reset_instance()
    asyncio.run(_run())
    print("[OK] E2E get_system_info OK")


def test_e2e_no_tool_needed():
    """E2E: question sans outil nécessaire → réponse directe sans tool_call."""
    if _skip_if_no_model():
        print("[SKIP] test_e2e_no_tool_needed (MODEL_PATH non défini)")
        return

    model_path = os.environ["MODEL_PATH"]

    async def _run():
        from tools.builtin import register_builtin_tools

        ToolRegistry._reset_instance()
        with tempfile.TemporaryDirectory() as tmp:
            reg = ToolRegistry(storage_path=Path(tmp) / "tools")
            await register_builtin_tools(reg)
            tools = reg.get_tools_for_ministral()

            engine = MinistralEngine(model_path)
            await engine.load()

            messages = [{"role": "user", "content": "Combien font 2 + 2 ?"}]
            events = []
            async for event in engine.stream_with_tools(messages, tools, reg, temperature=0.15):
                events.append(event)

            print(f"  Events: {[e['type'] for e in events]}")
            assert "final" in {e["type"] for e in events}

            ToolRegistry._reset_instance()
    asyncio.run(_run())
    print("[OK] E2E réponse directe sans outil OK")


# ═══════════════════════════════════════════════════════════════════════════════
# Performance
# ═══════════════════════════════════════════════════════════════════════════════


def test_perf_tool_injection_overhead():
    """Injection tools JSON < 50ms."""
    tools_sample = [s.to_openai_format() for _, (s, _) in BUILTIN_TOOLS.items()]

    t0 = time.perf_counter()
    for _ in range(1000):
        json.dumps(tools_sample)
    elapsed_ms = (time.perf_counter() - t0) / 1000 * 1000

    assert elapsed_ms < 50, f"Injection overhead trop élevée: {elapsed_ms:.3f}ms"
    print(f"[OK] Tool injection overhead: {elapsed_ms:.3f}ms avg (< 50ms)")


def test_perf_parsing_overhead():
    """Parsing tool calls JSON < 50ms par appel."""
    responses = [
        json.dumps({"tool_calls": [{"name": "get_current_time", "arguments": {"timezone": "UTC"}}]}),
        "[TOOL_CALLS] [{\"name\": \"get_system_info\", \"arguments\": {}}]",
        "Laisse-moi... " + json.dumps({"tool_calls": [{"name": "ping_host", "arguments": {"host": "8.8.8.8"}}]}) + " voilà",
        "Simple text without tools.",
    ]

    t0 = time.perf_counter()
    N = 10000
    for _ in range(N):
        for r in responses:
            MinistralEngine._parse_tool_calls(r)
    elapsed_ms = (time.perf_counter() - t0) / (N * len(responses)) * 1000

    assert elapsed_ms < 50, f"Parsing overhead trop élevé: {elapsed_ms:.4f}ms"
    print(f"[OK] Parsing overhead: {elapsed_ms:.4f}ms avg (< 50ms)")


# ═══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3.1 — Tool Calling Foundation — Tests")
    print("=" * 70)

    print("\n── 3.1.1 Tool Schema ──────────────────────────────────────────────")
    test_schema_openai_format_exact()
    test_schema_name_snake_case()
    test_schema_json_types_strict()
    test_schema_datetime_round_trip()
    test_schema_naive_datetime_rejected()
    test_schema_enum_in_output()
    test_schema_duplicate_param_names()

    print("\n── 3.1.2 Tool Registry ────────────────────────────────────────────")
    test_registry_singleton()
    test_registry_register_persist()
    test_registry_force_flag()
    test_registry_system_tool_protection()
    test_registry_openai_format_strict()
    test_registry_permission_filter()
    test_registry_persistence_reload()
    test_registry_concurrent_safety()
    test_registry_update_stats()

    print("\n── 3.1.3 Outils Builtin ───────────────────────────────────────────")
    test_builtin_get_current_time_valid()
    test_builtin_get_current_time_invalid_tz()
    test_builtin_get_system_info()
    test_builtin_ping_valid()
    test_builtin_ping_invalid()
    test_builtin_auto_registration()
    test_builtin_openai_schema_valid()

    print("\n── 3.1.4 MinistralEngine stream_with_tools ────────────────────────")
    test_engine_text_response_no_tools()
    test_engine_single_tool_call()
    test_engine_multi_tool_calls()
    test_engine_parsing_mixed_text()
    test_engine_tool_calls_token_format()
    test_engine_unknown_tool_graceful()
    test_engine_max_iterations()
    test_engine_all_event_types()
    test_engine_backward_compat()
    test_engine_model_not_loaded()

    print("\n── Performance ────────────────────────────────────────────────────")
    test_perf_tool_injection_overhead()
    test_perf_parsing_overhead()

    print("\n── 3.1.5 E2E (modèle réel) ────────────────────────────────────────")
    test_e2e_simple_tool_call()
    test_e2e_system_info()
    test_e2e_no_tool_needed()

    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS PHASE 3.1 PASSÉS")
    print("=" * 70)
