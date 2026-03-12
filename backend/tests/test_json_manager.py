"""
Tests de validation pour storage/json_manager.py.

Couvre les cas :
  - Dossier DATA_FOLDER inexistant → créé automatiquement
  - conversations.json absent → retourne []
  - conversations.json JSON invalide → gestion gracieuse
  - Auto-cleanup : fichier {uuid}.json manquant supprimé de l'index
  - Auto-cleanup : conversations.json réécrit avec entrées valides seulement
  - Datetime ISO 8601 avec timezone UTC
  - CRUD complet (save/load/delete conversation + index)
  - verify_integrity()

Exécution :
  cd backend && python -m pytest tests/test_json_manager.py -v
"""

import json
import sys
from datetime import timezone
from pathlib import Path

import pytest

# Ajouter backend/ au path (relatif à ce fichier)
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.conversation import Conversation, ConversationMetadata, Message, Role
from storage.json_manager import JSONManager


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_manager(tmp_path: Path) -> JSONManager:
    """JSONManager avec dossier temporaire vide."""
    return JSONManager(tmp_path / "conversations")


def _make_meta(conv_id: str, title: str = "Test") -> ConversationMetadata:
    return ConversationMetadata(id=conv_id, title=title, systemPrompt="Tu es un assistant.")


def _make_conv(conv_id: str) -> Conversation:
    conv = Conversation(id=conv_id, systemPrompt="Tu es un assistant.")
    conv.messages.append(Message(role=Role.user, content="Bonjour"))
    conv.messages.append(Message(role=Role.assistant, content="Bonjour !"))
    return conv


# ── Tests création dossier ────────────────────────────────────────────────────

def test_data_folder_created_if_missing(tmp_path: Path) -> None:
    """DATA_FOLDER inexistant → créé automatiquement à l'init."""
    folder = tmp_path / "nested" / "deep" / "conversations"
    assert not folder.exists()
    JSONManager(folder)
    assert folder.exists(), "Le dossier doit être créé automatiquement"


# ── Tests index absent / invalide ─────────────────────────────────────────────

def test_load_index_returns_empty_when_no_file(tmp_manager: JSONManager) -> None:
    """conversations.json absent → retourne liste vide (pas d'erreur)."""
    result = tmp_manager.load_index()
    assert result == []


def test_load_index_returns_empty_on_invalid_json(tmp_manager: JSONManager) -> None:
    """conversations.json JSON invalide → gestion gracieuse, retourne []."""
    index_path = tmp_manager.data_folder / "conversations.json"
    index_path.write_text("{not valid json", encoding="utf-8")
    result = tmp_manager.load_index()
    assert result == []


def test_load_index_returns_empty_on_non_list_json(tmp_manager: JSONManager) -> None:
    """conversations.json contient un objet (pas une liste) → retourne []."""
    index_path = tmp_manager.data_folder / "conversations.json"
    index_path.write_text('{"key": "value"}', encoding="utf-8")
    result = tmp_manager.load_index()
    assert result == []


# ── Tests CRUD index ──────────────────────────────────────────────────────────

def test_save_and_load_index(tmp_manager: JSONManager) -> None:
    """save_index puis load_index retourne les mêmes métadonnées."""
    meta1 = _make_meta("uuid-001", "Conversation A")
    meta2 = _make_meta("uuid-002", "Conversation B")

    # Créer les fichiers .json correspondants pour éviter le cleanup
    (tmp_manager.data_folder / "uuid-001.json").write_text("{}", encoding="utf-8")
    (tmp_manager.data_folder / "uuid-002.json").write_text("{}", encoding="utf-8")

    tmp_manager.save_index([meta1, meta2])
    result = tmp_manager.load_index()

    assert len(result) == 2
    assert result[0].id == "uuid-001"
    assert result[1].id == "uuid-002"
    assert result[0].title == "Conversation A"


# ── Tests auto-cleanup ────────────────────────────────────────────────────────

def test_auto_cleanup_removes_orphan_entries(tmp_manager: JSONManager) -> None:
    """
    Scénario validation principale :
    - 3 entrées dans conversations.json
    - 1 fichier {uuid}.json supprimé manuellement
    - load_index() retourne 2 entrées, logue warning
    - conversations.json réécrit avec 2 entrées
    """
    ids = ["aaa-111", "bbb-222", "ccc-333"]

    # Créer les 3 fichiers .json
    for conv_id in ids:
        (tmp_manager.data_folder / f"{conv_id}.json").write_text("{}", encoding="utf-8")

    # Sauvegarder l'index avec 3 entrées
    tmp_manager.save_index([_make_meta(i) for i in ids])

    # Simuler suppression manuelle du 2ème fichier
    (tmp_manager.data_folder / "bbb-222.json").unlink()

    # load_index() doit retourner 2 entrées
    result = tmp_manager.load_index()
    assert len(result) == 2
    assert all(m.id != "bbb-222" for m in result)

    # conversations.json doit avoir été réécrit avec 2 entrées
    raw = json.loads((tmp_manager.data_folder / "conversations.json").read_text())
    assert len(raw) == 2
    assert all(m["id"] != "bbb-222" for m in raw)


def test_auto_cleanup_5_orphans_on_100(tmp_manager: JSONManager) -> None:
    """5 entrées orphelines sur 100 → 95 entrées retournées, index réécrit."""
    all_ids = [f"conv-{i:03d}" for i in range(100)]
    orphan_ids = set(f"conv-{i:03d}" for i in range(5))

    for conv_id in all_ids:
        (tmp_manager.data_folder / f"{conv_id}.json").write_text("{}", encoding="utf-8")

    tmp_manager.save_index([_make_meta(i) for i in all_ids])

    # Supprimer 5 fichiers manuellement
    for conv_id in orphan_ids:
        (tmp_manager.data_folder / f"{conv_id}.json").unlink()

    result = tmp_manager.load_index()
    assert len(result) == 95
    assert all(m.id not in orphan_ids for m in result)

    raw = json.loads((tmp_manager.data_folder / "conversations.json").read_text())
    assert len(raw) == 95


def test_no_rewrite_when_no_orphans(tmp_manager: JSONManager) -> None:
    """Pas d'entrées orphelines → conversations.json non réécrit inutilement."""
    meta = _make_meta("no-orphan-id")
    (tmp_manager.data_folder / "no-orphan-id.json").write_text("{}", encoding="utf-8")
    tmp_manager.save_index([meta])

    # Récupérer mtime avant load
    index_path = tmp_manager.data_folder / "conversations.json"
    mtime_before = index_path.stat().st_mtime

    tmp_manager.load_index()

    mtime_after = index_path.stat().st_mtime
    assert mtime_before == mtime_after, "conversations.json ne doit pas être réécrit si pas d'orphelins"


# ── Tests CRUD conversation ───────────────────────────────────────────────────

def test_save_and_load_conversation(tmp_manager: JSONManager) -> None:
    """save_conversation puis load_conversation retourne la même structure."""
    conv = _make_conv("test-conv-id")
    tmp_manager.save_conversation(conv)

    loaded = tmp_manager.load_conversation("test-conv-id")
    assert loaded.id == "test-conv-id"
    assert loaded.systemPrompt == "Tu es un assistant."
    assert len(loaded.messages) == 2
    assert loaded.messages[0].role == Role.user
    assert loaded.messages[0].content == "Bonjour"


def test_load_conversation_raises_if_missing(tmp_manager: JSONManager) -> None:
    """load_conversation sur ID inexistant → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        tmp_manager.load_conversation("nonexistent-id")


def test_delete_conversation(tmp_manager: JSONManager) -> None:
    """delete_conversation supprime le fichier et l'entrée d'index."""
    conv = _make_conv("to-delete")
    meta = _make_meta("to-delete")
    meta2 = _make_meta("keep-me")
    (tmp_manager.data_folder / "keep-me.json").write_text("{}", encoding="utf-8")

    tmp_manager.save_conversation(conv)
    tmp_manager.save_index([meta, meta2])

    tmp_manager.delete_conversation("to-delete")

    # Fichier supprimé
    assert not (tmp_manager.data_folder / "to-delete.json").exists()

    # Index mis à jour
    index = tmp_manager.load_index()
    assert len(index) == 1
    assert index[0].id == "keep-me"


def test_delete_conversation_idempotent(tmp_manager: JSONManager) -> None:
    """delete_conversation sur ID inexistant → pas d'erreur."""
    tmp_manager.delete_conversation("ghost-id")  # Ne doit pas lever d'exception


# ── Tests datetime ISO 8601 ───────────────────────────────────────────────────

def test_datetime_iso8601_with_timezone(tmp_manager: JSONManager) -> None:
    """Les datetime sont sérialisés en ISO 8601 avec timezone (UTC)."""
    meta = _make_meta("datetime-test")
    (tmp_manager.data_folder / "datetime-test.json").write_text("{}", encoding="utf-8")
    tmp_manager.save_index([meta])

    raw = json.loads((tmp_manager.data_folder / "conversations.json").read_text())
    created_at = raw[0]["createdAt"]

    # Doit contenir +00:00 ou Z (timezone UTC explicite)
    assert "+" in created_at or created_at.endswith("Z"), (
        f"Timezone manquant dans createdAt: {created_at}"
    )
    # Doit contenir T (séparateur ISO 8601)
    assert "T" in created_at, f"Format ISO 8601 invalide: {created_at}"


def test_message_datetime_has_timezone(tmp_manager: JSONManager) -> None:
    """Les timestamp des messages ont un timezone."""
    conv = _make_conv("msg-datetime-test")
    tmp_manager.save_conversation(conv)

    raw = json.loads((tmp_manager.data_folder / "msg-datetime-test.json").read_text())
    ts = raw["messages"][0]["timestamp"]
    assert "T" in ts
    assert "+" in ts or ts.endswith("Z"), f"Timezone manquant dans message.timestamp: {ts}"


# ── Tests verify_integrity ────────────────────────────────────────────────────

def test_verify_integrity_returns_cleaned_count(tmp_manager: JSONManager) -> None:
    """verify_integrity retourne le bon nombre d'entrées nettoyées."""
    ids = ["v-aaa", "v-bbb", "v-ccc"]
    for conv_id in ids:
        (tmp_manager.data_folder / f"{conv_id}.json").write_text("{}", encoding="utf-8")
    tmp_manager.save_index([_make_meta(i) for i in ids])

    # Supprimer 2 fichiers
    (tmp_manager.data_folder / "v-bbb.json").unlink()
    (tmp_manager.data_folder / "v-ccc.json").unlink()

    result = tmp_manager.verify_integrity()
    assert result == {"cleaned": 2}


def test_verify_integrity_zero_when_clean(tmp_manager: JSONManager) -> None:
    """verify_integrity retourne cleaned=0 si tout est cohérent."""
    (tmp_manager.data_folder / "clean-id.json").write_text("{}", encoding="utf-8")
    tmp_manager.save_index([_make_meta("clean-id")])

    result = tmp_manager.verify_integrity()
    assert result == {"cleaned": 0}
