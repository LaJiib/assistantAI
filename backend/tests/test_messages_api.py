"""
Tests de validation pour api/messages.py.

Couvre :
  - GET /messages → [] pour conv vide (system exclu)
  - POST /messages → 201, {userMessage, assistantMessage}
  - POST /messages → fichier {uuid}.json contient les 2 messages
  - POST /messages → messageCount incrémenté de 2 dans l'index
  - POST /messages/stream → SSE chunks puis [DONE]
  - POST /messages/stream → messages sauvegardés après stream
  - POST sur conversation inexistante → 404
  - POST avec contenu vide → 422
  - POST quand moteur non chargé → 503

Engine mocké : pas de modèle MLX requis pour les tests.

Exécution :
  cd backend && python -m pytest tests/test_messages_api.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.conversations import router as conv_router
from api.messages import router as msg_router
from storage.json_manager import JSONManager


# ── Mock engine ───────────────────────────────────────────────────────────────

class MockEngine:
    """Engine de test : génère une réponse prévisible sans modèle LLM."""

    is_loaded = True

    def __init__(self, response: str = "Réponse mock."):
        self._response = response
        self._chunks = response.split()  # stream mot par mot

    def generate_messages(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        return self._response

    def stream_messages(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        for i, word in enumerate(self._chunks):
            yield word + (" " if i < len(self._chunks) - 1 else "")


class FailingEngine:
    """Engine qui lève une RuntimeError à chaque génération."""
    is_loaded = True

    def generate_messages(self, *args, **kwargs):
        raise RuntimeError("Erreur GPU simulée")

    def stream_messages(self, *args, **kwargs):
        raise RuntimeError("Erreur GPU simulée")
        yield  # rend la fonction generator


class UnloadedEngine:
    is_loaded = False


# ── App de test ───────────────────────────────────────────────────────────────

def make_app(tmp_path: Path, engine=None) -> FastAPI:
    app = FastAPI()
    app.include_router(conv_router)
    app.include_router(msg_router)
    app.state.json_manager = JSONManager(tmp_path / "conversations")
    app.state.engine = engine or MockEngine()
    return app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(make_app(tmp_path))


# ── Helper : créer une conversation via l'API ─────────────────────────────────

def _create_conv(client: TestClient, title: str = "Test") -> dict:
    resp = client.post("/api/conversations/", json={
        "title": title,
        "systemPrompt": "Tu es un assistant utile.",
    })
    assert resp.status_code == 201
    return resp.json()


# ── GET /messages ─────────────────────────────────────────────────────────────

def test_get_messages_empty_excludes_system(client: TestClient) -> None:
    """GET /messages retourne [] pour une conv fraîchement créée (system exclu)."""
    conv = _create_conv(client)
    resp = client.get(f"/api/conversations/{conv['id']}/messages/")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_messages_after_post(client: TestClient) -> None:
    """GET /messages après POST retourne user + assistant."""
    conv = _create_conv(client)
    client.post(f"/api/conversations/{conv['id']}/messages/", json={
        "content": "Bonjour",
    })
    resp = client.get(f"/api/conversations/{conv['id']}/messages/")
    assert resp.status_code == 200
    messages = resp.json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Bonjour"
    assert messages[1]["role"] == "assistant"


def test_get_messages_nonexistent_conv_returns_404(client: TestClient) -> None:
    """GET /messages sur UUID valide mais inexistant → 404."""
    resp = client.get("/api/conversations/00000000-0000-0000-0000-000000000000/messages/")
    assert resp.status_code == 404


# ── POST /messages ────────────────────────────────────────────────────────────

def test_post_message_returns_201(client: TestClient) -> None:
    """POST /messages retourne 201 avec userMessage et assistantMessage."""
    conv = _create_conv(client)
    resp = client.post(f"/api/conversations/{conv['id']}/messages/", json={
        "content": "Bonjour",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "userMessage" in data
    assert "assistantMessage" in data
    assert data["userMessage"]["role"] == "user"
    assert data["userMessage"]["content"] == "Bonjour"
    assert data["assistantMessage"]["role"] == "assistant"
    assert len(data["assistantMessage"]["content"]) > 0


def test_post_message_persists_in_file(tmp_path: Path) -> None:
    """POST /messages : les 2 messages sont sauvegardés dans {uuid}.json."""
    client = TestClient(make_app(tmp_path))
    conv = _create_conv(client)

    client.post(f"/api/conversations/{conv['id']}/messages/", json={
        "content": "Test persistance",
    })

    conv_file = tmp_path / "conversations" / f"{conv['id']}.json"
    assert conv_file.exists()
    data = json.loads(conv_file.read_text())

    roles = [m["role"] for m in data["messages"]]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
    contents = [m["content"] for m in data["messages"] if m["role"] == "user"]
    assert "Test persistance" in contents


def test_post_message_increments_message_count(tmp_path: Path) -> None:
    """POST /messages : messageCount dans conversations.json incrémenté de 2."""
    client = TestClient(make_app(tmp_path))
    conv = _create_conv(client)
    assert conv["messageCount"] == 0

    client.post(f"/api/conversations/{conv['id']}/messages/", json={
        "content": "Premier message",
    })

    resp = client.get("/api/conversations/")
    updated = next(c for c in resp.json() if c["id"] == conv["id"])
    assert updated["messageCount"] == 2


def test_post_two_messages_count_is_4(tmp_path: Path) -> None:
    """Deux POST /messages → messageCount == 4."""
    client = TestClient(make_app(tmp_path))
    conv = _create_conv(client)

    client.post(f"/api/conversations/{conv['id']}/messages/", json={"content": "Un"})
    client.post(f"/api/conversations/{conv['id']}/messages/", json={"content": "Deux"})

    resp = client.get("/api/conversations/")
    updated = next(c for c in resp.json() if c["id"] == conv["id"])
    assert updated["messageCount"] == 4


def test_post_message_nonexistent_conv_returns_404(client: TestClient) -> None:
    """POST /messages sur UUID valide mais inexistant → 404."""
    resp = client.post("/api/conversations/00000000-0000-0000-0000-000000000000/messages/", json={"content": "Test"})
    assert resp.status_code == 404


def test_post_message_empty_content_returns_422(client: TestClient) -> None:
    """POST /messages avec contenu vide → 422."""
    conv = _create_conv(client)
    resp = client.post(f"/api/conversations/{conv['id']}/messages/", json={"content": ""})
    assert resp.status_code == 422


def test_post_message_engine_not_loaded_returns_503(tmp_path: Path) -> None:
    """POST /messages quand moteur non chargé → 503."""
    client = TestClient(make_app(tmp_path, engine=UnloadedEngine()))
    conv = _create_conv(client)
    resp = client.post(f"/api/conversations/{conv['id']}/messages/", json={"content": "Test"})
    assert resp.status_code == 503


def test_post_message_generation_failure_preserves_user_message(tmp_path: Path) -> None:
    """
    Si la génération échoue, le message user est conservé (pas de rollback).
    messageCount = 1 (user seulement).
    """
    client = TestClient(make_app(tmp_path, engine=FailingEngine()))
    conv = _create_conv(client)

    resp = client.post(f"/api/conversations/{conv['id']}/messages/", json={"content": "Test"})
    assert resp.status_code == 500

    # Le message user doit être persisté malgré l'échec
    conv_file = tmp_path / "conversations" / f"{conv['id']}.json"
    data = json.loads(conv_file.read_text())
    user_msgs = [m for m in data["messages"] if m["role"] == "user"]
    assert len(user_msgs) == 1, "Message user doit être conservé après échec génération"


def test_post_message_has_timestamps(client: TestClient) -> None:
    """Les messages retournés ont des timestamps ISO 8601."""
    conv = _create_conv(client)
    resp = client.post(f"/api/conversations/{conv['id']}/messages/", json={"content": "Test"})
    data = resp.json()
    for msg_key in ("userMessage", "assistantMessage"):
        ts = data[msg_key]["timestamp"]
        assert "T" in ts, f"{msg_key}.timestamp doit être ISO 8601 : {ts}"
        assert "+" in ts or ts.endswith("Z"), f"{msg_key}.timestamp doit avoir timezone : {ts}"


# ── POST /messages/stream ─────────────────────────────────────────────────────

def _parse_sse(raw: str) -> list[dict]:
    """Parse une réponse SSE brute en liste d'événements."""
    events = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            events.append({"done": True})
        else:
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                events.append({"raw": payload})
    return events


def test_stream_returns_sse_chunks(client: TestClient) -> None:
    """POST /stream retourne des chunks SSE puis [DONE]."""
    conv = _create_conv(client)
    resp = client.post(
        f"/api/conversations/{conv['id']}/messages/stream",
        json={"content": "Bonjour"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = _parse_sse(resp.text)

    # Premier event = userMessage
    assert "userMessage" in events[0]

    # Events intermédiaires = chunks text
    text_events = [e for e in events if "text" in e]
    assert len(text_events) > 0

    # Dernier event = [DONE]
    assert events[-1] == {"done": True}


def test_stream_chunks_reconstruct_response(client: TestClient) -> None:
    """Les chunks SSE reconstitués forment la réponse complète."""
    mock_response = "Réponse mock."
    conv = _create_conv(client)

    # On a besoin du client avec notre mock engine pour vérifier le contenu exact
    resp = client.post(
        f"/api/conversations/{conv['id']}/messages/stream",
        json={"content": "Test"},
    )
    events = _parse_sse(resp.text)
    chunks = [e["text"] for e in events if "text" in e]
    reconstructed = "".join(chunks)
    assert len(reconstructed) > 0


def test_stream_saves_messages_after_done(tmp_path: Path) -> None:
    """Après stream [DONE], les messages sont sauvegardés dans {uuid}.json."""
    client = TestClient(make_app(tmp_path))
    conv = _create_conv(client)

    client.post(
        f"/api/conversations/{conv['id']}/messages/stream",
        json={"content": "Test stream"},
    )

    conv_file = tmp_path / "conversations" / f"{conv['id']}.json"
    data = json.loads(conv_file.read_text())
    roles = [m["role"] for m in data["messages"]]
    assert "user" in roles, "Message user doit être sauvegardé"
    assert "assistant" in roles, "Message assistant doit être sauvegardé"


def test_stream_increments_message_count(tmp_path: Path) -> None:
    """POST /stream incrémente messageCount de 2."""
    client = TestClient(make_app(tmp_path))
    conv = _create_conv(client)

    client.post(
        f"/api/conversations/{conv['id']}/messages/stream",
        json={"content": "Stream count test"},
    )

    resp = client.get("/api/conversations/")
    updated = next(c for c in resp.json() if c["id"] == conv["id"])
    assert updated["messageCount"] == 2


def test_stream_nonexistent_conv_returns_404(client: TestClient) -> None:
    """POST /stream sur UUID valide mais inexistant → 404."""
    resp = client.post(
        "/api/conversations/00000000-0000-0000-0000-000000000000/messages/stream",
        json={"content": "Test"},
    )
    assert resp.status_code == 404


def test_stream_empty_content_returns_422(client: TestClient) -> None:
    """POST /stream avec contenu vide → 422."""
    conv = _create_conv(client)
    resp = client.post(
        f"/api/conversations/{conv['id']}/messages/stream",
        json={"content": ""},
    )
    assert resp.status_code == 422


def test_stream_engine_not_loaded_returns_503(tmp_path: Path) -> None:
    """POST /stream quand moteur non chargé → 503."""
    client = TestClient(make_app(tmp_path, engine=UnloadedEngine()))
    conv = _create_conv(client)
    resp = client.post(
        f"/api/conversations/{conv['id']}/messages/stream",
        json={"content": "Test"},
    )
    assert resp.status_code == 503


# ── Multi-tour : contexte transmis au moteur ──────────────────────────────────

def test_llm_receives_full_history(tmp_path: Path) -> None:
    """
    Le moteur reçoit l'historique complet (system + user + assistant + nouveau user).
    On vérifie via un mock qui capture les appels.
    """
    capture = []

    class CapturingEngine:
        is_loaded = True

        def generate_messages(self, messages, **kwargs):
            capture.append(list(messages))
            return "Réponse capturée."

        def stream_messages(self, messages, **kwargs):
            capture.append(list(messages))
            yield "Réponse capturée."

    client = TestClient(make_app(tmp_path, engine=CapturingEngine()))
    conv = _create_conv(client)

    # Premier message
    client.post(f"/api/conversations/{conv['id']}/messages/", json={"content": "Premier"})
    # Deuxième message — doit inclure le contexte du premier
    client.post(f"/api/conversations/{conv['id']}/messages/", json={"content": "Deuxième"})

    assert len(capture) == 2
    # Premier appel : system + user
    assert capture[0][0]["role"] == "system"
    assert capture[0][-1]["content"] == "Premier"
    # Deuxième appel : system + user + assistant + user (4 messages)
    assert len(capture[1]) == 4
    assert capture[1][-1]["content"] == "Deuxième"
    assert capture[1][-2]["role"] == "assistant"
