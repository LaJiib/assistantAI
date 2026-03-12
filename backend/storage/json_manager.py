"""
JSONManager — Lecture/écriture des fichiers JSON de conversations.

Layout fichiers :
  data_folder/
    conversations.json     ← liste de ConversationMetadata
    {uuid}.json            ← objet Conversation avec messages[]

Thread-safety : threading.Lock protège toutes les opérations I/O
(single-process FastAPI + uvicorn, multi-thread via thread pool).

Auto-cleanup : load_index() supprime silencieusement les entrées de
conversations.json dont le {uuid}.json correspondant est absent.
Protection : le cleanup n'est déclenché que si data_folder est
accessible, pour éviter de nettoyer lors d'un démontage temporaire
du SSD externe.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List

from models.conversation import Conversation, ConversationMetadata

logger = logging.getLogger(__name__)

_INDEX_FILE = "conversations.json"


class JSONManager:
    def __init__(self, data_folder: Path) -> None:
        self.data_folder = Path(data_folder)
        self._lock = threading.Lock()
        self._ensure_data_folder()

    # ── Dossier ───────────────────────────────────────────────────────────────

    def _ensure_data_folder(self) -> None:
        """Crée data_folder si absent (ex: premier démarrage)."""
        self.data_folder.mkdir(parents=True, exist_ok=True)

    def _index_path(self) -> Path:
        return self.data_folder / _INDEX_FILE

    def _conversation_path(self, conv_id: str) -> Path:
        return self.data_folder / f"{conv_id}.json"

    # ── Lecture index brute (sans lock) ───────────────────────────────────────

    def _read_index_file(self) -> List[ConversationMetadata]:
        """
        Lit conversations.json et retourne la liste de métadonnées.

        Cas gérés :
          - Fichier absent       → retourne []  (premier démarrage normal)
          - JSON invalide        → log error + retourne []  (gracieux)
          - Erreur I/O           → log error + retourne []
        """
        path = self._index_path()
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, list):
                logger.error(
                    "conversations.json : format invalide (attendu liste JSON), "
                    "réinitialisation à []"
                )
                return []
            return [ConversationMetadata.model_validate(item) for item in data]
        except json.JSONDecodeError as exc:
            logger.error("conversations.json : JSON invalide (%s), réinitialisation à []", exc)
            return []
        except Exception as exc:
            logger.error("conversations.json : erreur lecture (%s), réinitialisation à []", exc)
            return []

    # ── API publique ──────────────────────────────────────────────────────────

    def load_index(self) -> List[ConversationMetadata]:
        """
        Charge l'index avec auto-cleanup des entrées orphelines.

        Une entrée est orpheline si son {uuid}.json est absent ET que
        data_folder est accessible (protection contre SSD démonté).
        Si des entrées sont nettoyées, conversations.json est réécrit
        immédiatement pour garantir la cohérence à la prochaine lecture.
        """
        with self._lock:
            # Protection : ne pas cleanup si le dossier est inaccessible
            if not self.data_folder.exists():
                logger.error(
                    "data_folder inaccessible (%s), auto-cleanup suspendu",
                    self.data_folder,
                )
                return []

            index = self._read_index_file()
            if not index:
                return []

            valid: List[ConversationMetadata] = []
            orphans: List[str] = []

            for meta in index:
                if self._conversation_path(meta.id).exists():
                    valid.append(meta)
                else:
                    orphans.append(meta.id)

            if orphans:
                for conv_id in orphans:
                    logger.warning(
                        "Auto-cleanup : %s.json manquant, supprimé de l'index", conv_id
                    )
                # Réécriture immédiate → cohérence garantie pour les prochaines lectures
                self._write_index_file(valid)

            return valid

    def save_index(self, metadata: List[ConversationMetadata]) -> None:
        with self._lock:
            self._write_index_file(metadata)

    def load_conversation(self, conv_id: str) -> Conversation:
        """
        Charge {uuid}.json.

        Raises:
            FileNotFoundError : si le fichier est absent.
            ValueError        : si le JSON est invalide.
        """
        with self._lock:
            path = self._conversation_path(conv_id)
            if not path.exists():
                raise FileNotFoundError(f"{conv_id}.json introuvable dans {self.data_folder}")
            try:
                raw = path.read_text(encoding="utf-8")
                return Conversation.model_validate_json(raw)
            except Exception as exc:
                raise ValueError(f"{conv_id}.json invalide : {exc}") from exc

    def save_conversation(self, conversation: Conversation) -> None:
        with self._lock:
            path = self._conversation_path(conversation.id)
            path.write_text(
                conversation.model_dump_json(indent=2),
                encoding="utf-8",
            )

    def delete_conversation(self, conv_id: str) -> None:
        """
        Supprime {uuid}.json et retire l'entrée de conversations.json.
        Idempotent : pas d'erreur si déjà absent.
        """
        with self._lock:
            # Supprimer le fichier conversation
            path = self._conversation_path(conv_id)
            if path.exists():
                path.unlink()

            # Retirer de l'index
            index = self._read_index_file()
            updated = [m for m in index if m.id != conv_id]
            if len(updated) < len(index):
                self._write_index_file(updated)

    def verify_integrity(self) -> Dict[str, int]:
        """
        Vérifie la cohérence index ↔ fichiers.
        Retourne {"cleaned": N} avec N = nombre d'entrées supprimées.

        Appelé au startup depuis main.py pour un cleanup initial.
        Réutilise load_index() qui effectue déjà le cleanup — on compte
        simplement la différence avant/après.
        """
        with self._lock:
            raw_index = self._read_index_file()
            n_before = len(raw_index)

        # load_index() prend son propre lock — on appelle sans lock ici
        clean_index = self.load_index()
        cleaned = n_before - len(clean_index)

        if cleaned > 0:
            logger.info("verify_integrity : %d entrée(s) orpheline(s) nettoyée(s)", cleaned)
        else:
            logger.debug("verify_integrity : index cohérent (%d conversations)", len(clean_index))

        return {"cleaned": cleaned}

    # ── Helpers privés ────────────────────────────────────────────────────────

    def _write_index_file(self, metadata: List[ConversationMetadata]) -> None:
        """Écrit conversations.json (supposé appelé sous lock)."""
        self._ensure_data_folder()
        data = [m.model_dump(mode="json") for m in metadata]
        self._index_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
