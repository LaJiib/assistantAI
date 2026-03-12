"""
Tests de validation pour api/conversations.py.

Couvre :
  - GET /api/conversations/       → liste vide, liste avec données
  - POST /api/conversations/      → 201, fichier créé, validation titre
  - GET /api/conversations/{id}   → 200 avec messages, 404 si inexistant
  - PUT /api/conversations/{id}   → 200, updatedAt modifié, createdAt immutable
  - DELETE /api/conversations/{id}→ 204, 404 sur ID inexistant
  - Tri par updatedAt décroissant
  - Datetime ISO 8601 dans réponses

Exécution :
  cd backend && python -m pytest tests/test_conversations_api.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.conversations import router
from storage.json_manager import JSONManager


# ── App de test ───────────────────────────────────────────────────────────────

def make_app(tmp_path: Path) -> FastAPI:
    """Crée une app FastAPI de test avec JSONManager sur tmp_path."""
    app = FastAPI()
    app.include_router(router)
    app.state.json_manager = JSONManager(tmp_path / "conversations")
    return app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(make_app(tmp_path))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _create_conv(client: TestClient, title: str = "Test", system: str = "Tu es un assistant.") -> dict:
    resp = client.post("/api/conversations/", json={"title": title, "systemPrompt": system})
    assert resp.status_code == 201
    return resp.json()


# ── GET / ─────────────────────────────────────────────────────────────────────

def test_list_empty(client: TestClient) -> None:
    """GET / retourne liste vide au premier démarrage."""
    resp = client.get("/api/conversations/")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_returns_created_conversations(client: TestClient) -> None:
    """GET / retourne toutes les conversations créées."""
    _create_conv(client, "Conversation A")
    _create_conv(client, "Conversation B")

    resp = client.get("/api/conversations/")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    titles = {c["title"] for c in data}
    assert titles == {"Conversation A", "Conversation B"}


def test_list_sorted_by_updated_at_desc(client: TestClient, tmp_path: Path) -> None:
    """GET / retourne les conversations triées par updatedAt décroissant."""
    conv_a = _create_conv(client, "Première")
    time.sleep(0.01)  # garantit un updatedAt différent
    conv_b = _create_conv(client, "Deuxième")

    resp = client.get("/api/conversations/")
    data = resp.json()
    assert data[0]["id"] == conv_b["id"], "La plus récente doit être en premier"
    assert data[1]["id"] == conv_a["id"]


# ── POST / ────────────────────────────────────────────────────────────────────

def test_create_returns_201(client: TestClient) -> None:
    """POST / retourne 201 avec les métadonnées."""
    resp = client.post("/api/conversations/", json={
        "title": "Ma conversation",
        "systemPrompt": "Tu es un assistant utile.",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Ma conversation"
    assert data["systemPrompt"] == "Tu es un assistant utile."
    assert data["messageCount"] == 0
    assert "id" in data
    assert "createdAt" in data
    assert "updatedAt" in data


def test_create_file_exists(client: TestClient, tmp_path: Path) -> None:
    """POST / crée le fichier {uuid}.json."""
    conv = _create_conv(client)
    conv_file = tmp_path / "conversations" / f"{conv['id']}.json"
    assert conv_file.exists(), f"{conv['id']}.json doit être créé"


def test_create_has_system_message(client: TestClient) -> None:
    """La conversation créée contient un message system initial."""
    conv = _create_conv(client, system="Prompt system test")

    resp = client.get(f"/api/conversations/{conv['id']}")
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "Prompt system test"


def test_create_empty_title_returns_422(client: TestClient) -> None:
    """POST avec titre vide → 422 (validation Pydantic)."""
    resp = client.post("/api/conversations/", json={
        "title": "",
        "systemPrompt": "Prompt.",
    })
    assert resp.status_code == 422


def test_create_title_too_long_returns_422(client: TestClient) -> None:
    """POST avec titre > 500 chars → 422."""
    resp = client.post("/api/conversations/", json={
        "title": "x" * 501,
        "systemPrompt": "Prompt.",
    })
    assert resp.status_code == 422


def test_create_empty_system_prompt_returns_422(client: TestClient) -> None:
    """POST avec systemPrompt vide → 422."""
    resp = client.post("/api/conversations/", json={
        "title": "Titre",
        "systemPrompt": "",
    })
    assert resp.status_code == 422


def test_create_missing_fields_returns_422(client: TestClient) -> None:
    """POST sans body → 422."""
    resp = client.post("/api/conversations/", json={})
    assert resp.status_code == 422


# ── GET /{id} ─────────────────────────────────────────────────────────────────

def test_get_conversation_returns_detail(client: TestClient) -> None:
    """GET /{id} retourne métadonnées + messages."""
    conv = _create_conv(client, "Détail test")

    resp = client.get(f"/api/conversations/{conv['id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == conv["id"]
    assert data["title"] == "Détail test"
    assert "messages" in data
    assert isinstance(data["messages"], list)


def test_get_nonexistent_returns_404(client: TestClient) -> None:
    """GET /{id} sur UUID valide mais inexistant → 404."""
    resp = client.get("/api/conversations/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_get_after_file_deleted_returns_404(client: TestClient, tmp_path: Path) -> None:
    """
    GET /{id} quand {uuid}.json supprimé manuellement → 404.
    load_index() auto-cleanup → l'ID disparaît de l'index.
    """
    conv = _create_conv(client)
    (tmp_path / "conversations" / f"{conv['id']}.json").unlink()

    resp = client.get(f"/api/conversations/{conv['id']}")
    assert resp.status_code == 404


# ── PUT /{id} ─────────────────────────────────────────────────────────────────

def test_update_title(client: TestClient) -> None:
    """PUT /{id} modifie le titre."""
    conv = _create_conv(client, "Ancien titre")

    resp = client.put(f"/api/conversations/{conv['id']}", json={"title": "Nouveau titre"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "Nouveau titre"


def test_update_title_updated_at_changes(client: TestClient) -> None:
    """PUT /{id} met à jour updatedAt."""
    conv = _create_conv(client)
    created_at = conv["createdAt"]
    updated_at_before = conv["updatedAt"]

    time.sleep(0.01)
    resp = client.put(f"/api/conversations/{conv['id']}", json={"title": "Nouveau"})
    data = resp.json()

    assert data["createdAt"] == created_at, "createdAt doit rester immutable"
    assert data["updatedAt"] > updated_at_before, "updatedAt doit être mis à jour"


def test_update_system_prompt_not_changed(client: TestClient) -> None:
    """PUT /{id} ne modifie pas systemPrompt."""
    conv = _create_conv(client, system="Prompt original")

    resp = client.put(f"/api/conversations/{conv['id']}", json={"title": "Nouveau titre"})
    assert resp.json()["systemPrompt"] == "Prompt original"


def test_update_nonexistent_returns_404(client: TestClient) -> None:
    """PUT /{id} sur UUID valide mais inexistant → 404."""
    resp = client.put("/api/conversations/00000000-0000-0000-0000-000000000000", json={"title": "Titre"})
    assert resp.status_code == 404


def test_update_empty_title_returns_422(client: TestClient) -> None:
    """PUT /{id} avec titre vide → 422."""
    conv = _create_conv(client)
    resp = client.put(f"/api/conversations/{conv['id']}", json={"title": ""})
    assert resp.status_code == 422


# ── DELETE /{id} ─────────────────────────────────────────────────────────────

def test_delete_returns_204(client: TestClient) -> None:
    """DELETE /{id} retourne 204."""
    conv = _create_conv(client)
    resp = client.delete(f"/api/conversations/{conv['id']}")
    assert resp.status_code == 204


def test_delete_removes_file(client: TestClient, tmp_path: Path) -> None:
    """DELETE /{id} supprime le fichier {uuid}.json."""
    conv = _create_conv(client)
    conv_file = tmp_path / "conversations" / f"{conv['id']}.json"
    assert conv_file.exists()

    client.delete(f"/api/conversations/{conv['id']}")
    assert not conv_file.exists()


def test_delete_removes_from_index(client: TestClient) -> None:
    """DELETE /{id} retire la conversation de la liste."""
    conv = _create_conv(client)
    client.delete(f"/api/conversations/{conv['id']}")

    resp = client.get("/api/conversations/")
    ids = [c["id"] for c in resp.json()]
    assert conv["id"] not in ids


def test_delete_nonexistent_returns_404(client: TestClient) -> None:
    """DELETE /{id} sur UUID valide mais inexistant → 404."""
    resp = client.delete("/api/conversations/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_delete_then_get_returns_404(client: TestClient) -> None:
    """Après DELETE, GET /{id} retourne 404."""
    conv = _create_conv(client)
    client.delete(f"/api/conversations/{conv['id']}")

    resp = client.get(f"/api/conversations/{conv['id']}")
    assert resp.status_code == 404


# ── Datetime ISO 8601 ─────────────────────────────────────────────────────────

def test_response_datetimes_are_iso8601(client: TestClient) -> None:
    """Les dates dans les réponses sont en ISO 8601 avec timezone."""
    conv = _create_conv(client)
    for field in ("createdAt", "updatedAt"):
        value = conv[field]
        assert "T" in value, f"{field} doit être ISO 8601 : {value}"
        assert "+" in value or value.endswith("Z"), f"{field} doit avoir un timezone : {value}"
