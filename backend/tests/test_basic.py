"""
Tests d'intégration du backend AssistantIA.
Nécessite que le serveur tourne : python main.py

Usage :
    python3 tests/test_basic.py
    BASE_URL=http://127.0.0.1:9000 python3 tests/test_basic.py
    python3 tests/test_basic.py --wait    # attend 30s le démarrage

Exit : 0 si tous les tests passent, 1 sinon.
"""

import json
import os
import sys
import time

try:
    import requests
    from requests.exceptions import ConnectionError as ReqConnError, Timeout
except ImportError:
    print("❌ 'requests' manquant : pip install requests")
    sys.exit(1)

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
PERF_MIN = 30   # tok/s minimum

_results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    icon = "✅" if passed else "❌"
    print(f"  {icon} {name}" + (f"  {detail}" if detail else ""))
    _results.append((name, passed, detail))


# ── Connexion ─────────────────────────────────────────────────────────────────

def check_reachable() -> None:
    if "--wait" in sys.argv:
        deadline = time.time() + 30
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                requests.get(f"{BASE_URL}/health", timeout=2)
                if attempt > 1:
                    print(f"\n  Backend prêt ({attempt}s)")
                return
            except (ReqConnError, Timeout):
                print("." if attempt > 1 else "  Attente backend (30s max).", end="", flush=True)
                time.sleep(1)
        print()

    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
        return
    except (ReqConnError, Timeout):
        pass

    print(f"\n❌ Backend inaccessible à {BASE_URL}")
    print("   Lancez d'abord : python main.py")
    sys.exit(1)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_health() -> bool:
    print("\n── /health ──────────────────────────────────")
    resp = requests.get(f"{BASE_URL}/health", timeout=5)

    ok_status = resp.status_code == 200
    record("GET /health → 200", ok_status, f"reçu {resp.status_code}" if not ok_status else "")
    if not ok_status:
        return False

    data = resp.json()
    has_fields = all(k in data for k in ("status", "model_loaded", "model_name"))
    record("champs présents", has_fields)

    loaded = bool(data.get("model_loaded"))
    record("model_loaded = true", loaded,
           data.get("model_name", "") if loaded else "modèle non chargé")
    return ok_status and has_fields and loaded


def test_chat() -> bool:
    print("\n── /chat ─────────────────────────────────────")
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"prompt": "Réponds en une phrase : qu'est-ce qu'un LLM ?", "max_tokens": 80},
        timeout=60,
    )
    ok = resp.status_code == 200
    record("POST /chat → 200", ok, f"reçu {resp.status_code}" if not ok else "")
    if not ok:
        return False

    text = resp.json().get("response", "")
    non_empty = isinstance(text, str) and len(text.strip()) >= 10
    record("réponse non vide", non_empty, f"{len(text)} chars")
    if non_empty:
        print(f"  ℹ️  «{text.strip()[:120]}»")
    return non_empty


def test_chat_stream() -> bool:
    print("\n── /chat/stream ──────────────────────────────")
    resp = requests.post(
        f"{BASE_URL}/chat/stream",
        json={"prompt": "Réponds en une phrase : qu'est-ce qu'un GPU ?", "max_tokens": 80},
        stream=True,
        timeout=60,
    )
    ok_status = resp.status_code == 200
    record("POST /chat/stream → 200", ok_status, f"reçu {resp.status_code}" if not ok_status else "")
    if not ok_status:
        return False

    full_text, chunks, done = "", 0, False
    for raw in resp.iter_lines():
        line = raw.decode() if isinstance(raw, bytes) else raw
        if line == "data: [DONE]":
            done = True
            break
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                if "error" in data:
                    record("pas d'erreur mid-stream", False, data["error"])
                    return False
                full_text += data.get("text", "")
                chunks += 1
            except json.JSONDecodeError:
                pass

    record("[DONE] reçu", done)
    record("chunks > 1", chunks > 1, f"{chunks} chunks")
    non_empty = len(full_text.strip()) >= 10
    record("texte assemblé non vide", non_empty, f"{len(full_text)} chars")
    if non_empty:
        print(f"  ℹ️  «{full_text.strip()[:120]}»")
    return done and non_empty


def test_performance() -> bool:
    print("\n── performance /chat/stream ──────────────────")
    print("  max_tokens=200, génération en cours…")

    t0 = time.time()
    resp = requests.post(
        f"{BASE_URL}/chat/stream",
        json={
            "prompt": "Explique en détail comment fonctionne un transformeur en IA.",
            "max_tokens": 200,
            "temperature": 0.3,
        },
        stream=True,
        timeout=120,
    )
    full_text = ""
    for raw in resp.iter_lines():
        line = raw.decode() if isinstance(raw, bytes) else raw
        if line == "data: [DONE]":
            break
        if line.startswith("data: "):
            try:
                full_text += json.loads(line[6:]).get("text", "")
            except json.JSONDecodeError:
                pass
    elapsed = time.time() - t0

    # ~4 chars/token en moyenne (texte technique français/anglais)
    approx_tokens = max(len(full_text) // 4, 1)
    tok_per_sec   = approx_tokens / elapsed

    print(f"  Temps   : {elapsed:.2f}s  |  Chars : {len(full_text)}  (~{approx_tokens} tokens)")
    print(f"  Débit   : ~{tok_per_sec:.1f} tok/s  (cible ≥ {PERF_MIN})")
    print(f"  ℹ️  Tok/s réels dans les logs serveur (generation_tps)")

    passed = tok_per_sec >= PERF_MIN
    record(f"performance ≥ {PERF_MIN} tok/s", passed, f"~{tok_per_sec:.1f} tok/s")
    return passed


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'═'*52}")
    print(f"  AssistantIA — Tests backend  {BASE_URL}")
    print(f"{'═'*52}")

    check_reachable()

    health_ok = test_health()
    if health_ok:
        test_chat()
        test_chat_stream()
        test_performance()

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed

    print(f"\n{'═'*52}")
    if failed == 0:
        print(f"  ✅ {passed}/{len(_results)} tests passés")
    else:
        print(f"  ❌ {failed}/{len(_results)} tests échoués")
        for name, ok, detail in _results:
            if not ok:
                print(f"     • {name}" + (f" : {detail}" if detail else ""))
    print(f"{'═'*52}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
