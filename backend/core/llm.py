"""
MinistralEngine — wrapper mlx-lm pour Ministral 3 14B Instruct.

Cycle de vie :
    engine = MinistralEngine(model_path)
    await engine.load()                       # startup FastAPI
    engine.stream(prompt, ...)                # sync, itère des chunks texte (compat Phase 2)
    engine.generate(prompt, ...)              # sync, retourne le texte complet (compat Phase 2)
    engine.stream_messages(messages, ...)     # sync, multi-turn via historique messages
    engine.generate_messages(messages, ...)   # sync, multi-turn, retourne texte complet
"""

import asyncio
import logging
from typing import Dict, Iterator, List

from mlx_lm import load as mlx_load, stream_generate
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger(__name__)


class MinistralEngine:

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_name(self) -> str:
        return self.model_path.rstrip("/").split("/")[-1]

    async def load(self) -> None:
        """Charge le modèle MLX au startup. Bloquant → isolé dans un executor."""
        if self._is_loaded:
            return
        loop = asyncio.get_running_loop()
        self._model, self._tokenizer = await loop.run_in_executor(
            None, mlx_load, self.model_path
        )
        self._is_loaded = True
        logger.info(f"✅ Modèle chargé : {self.model_name}")

    # ── Génération à partir d'un historique de messages ───────────────────────

    def stream_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """
        Génère le texte chunk par chunk depuis un historique de messages.

        Args:
            messages: liste de dicts {"role": "user"|"assistant"|"system", "content": str}
                      dans l'ordre chronologique. Le tokenizer applique le chat template
                      natif du modèle → support multi-turn correct.

        Synchrone — appelé depuis un endpoint FastAPI def (non-async).
        """
        if not self._is_loaded:
            raise RuntimeError("Modèle non chargé. Appelez await engine.load() d'abord.")

        formatted = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        last = None
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        ):
            yield response.text
            last = response

        if last:
            logger.info(
                f"{last.generation_tokens} tokens "
                f"@ {last.generation_tps:.1f} tok/s"
            )

    def generate_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Retourne le texte complet depuis un historique de messages."""
        return "".join(self.stream_messages(messages, max_tokens, temperature))

    # ── API compat Phase 2 (prompt texte brut) ────────────────────────────────

    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """
        Génère depuis un prompt texte brut (compat Phase 2 / /chat endpoint).
        Délègue à stream_messages() avec un message user unique.
        """
        return self.stream_messages(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """Retourne le texte complet (compat Phase 2)."""
        return "".join(self.stream(prompt, max_tokens, temperature))
