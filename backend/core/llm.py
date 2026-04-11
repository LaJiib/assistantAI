"""
IrisEngine — Moteur d'inférence MLX pour Iris.

Modèle cible : google/gemma-4-26b-a4b (Q8_0, ~28 Go sur 48 Go M4 Pro).

KV Cache B1 (System Prompt Pre-fill) :
  - Avantage   : ~200ms économisés par requête (recompute K/V system prompt évité).
  - Principe   : cache pré-rempli au load(), réutilisé à chaque génération.
  - Sécurité   : _cache_lock (threading.Lock) sérialise l'accès → pas de corruption.
    Acceptable car le GPU M4 Pro ne peut traiter qu'un seul stream à la fois.
  - Trim       : trim_prompt_cache() remet le cache au baseline entre requêtes.
  - Dégradation: si mlx_lm.cache indisponible → fallback génération standard.

Tool Calling :
  - stream_with_tools() — boucle agent async avec events typés.
  - _parse_tool_calls() — supporte format Mistral [TOOL_CALLS] et JSON object.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional

from mlx_vlm import load as mlx_load, stream_generate

# KV Cache B1 — désactivé : mlx-vlm utilise prompt_cache_state= (API différente de mlx-lm)
# TODO : ré-implémenter via mlx_vlm.models.cache.make_prompt_cache + prompt_cache_state=
_HAS_PROMPT_CACHE = False

logger = logging.getLogger(__name__)

# Token spécial Mistral/Tekken signalant un tool call dans la réponse
_TOOL_CALLS_TOKEN = "[TOOL_CALLS]"


class IrisEngine:
    """
    Moteur d'inférence MLX pour Iris.

    Cycle de vie :
        engine = IrisEngine(model_path)
        await engine.load()                        # startup — charge modèle + warm cache
        engine.stream_messages(messages, ...)      # streaming sync (FastAPI SSE)
        engine.generate_messages(messages, ...)    # génération complète sync
        await engine.stream_with_tools(...)        # agent loop async avec tool calling

    KV Cache B1 :
        Temporairement désactivé — mlx-vlm utilise prompt_cache_state= (API différente).
        TODO : ré-implémenter via mlx_vlm.models.cache.make_prompt_cache.
    """

    # System prompt par défaut d'Iris — pré-rempli dans le KV cache au chargement.
    # Pour les conversations avec system prompt custom, celui-ci est injecté via
    # le message system dans l'historique → sera traité par le chat template.
    IRIS_SYSTEM_PROMPT = (
        "Tu es Iris, une assistante de consulting stratégique haute performance. "
        "Tu opères en local sur un Mac Mini M4 Pro (48 Go RAM, 100% privé — aucune "
        "donnée ne quitte le système). Tu es analytique, précise et directe. "
        "Pour chaque requête complexe, tu raisonnes étape par étape avant de répondre."
    )

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None
        self._processor = None   # mlx-vlm processor (passé à stream_generate)
        self._tokenizer = None   # processor.tokenizer (pour apply_chat_template)
        self._is_loaded = False

        # KV Cache B1 — system prompt pre-fill
        self._prompt_cache: Optional[list] = None
        self._system_cache_len: int = 0     # tokens dans le baseline cache
        self._cache_lock = threading.Lock() # sérialise l'accès au cache partagé

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_name(self) -> str:
        return self.model_path.rstrip("/").split("/")[-1]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def load(self) -> None:
        """
        Charge le modèle MLX et pré-remplit le KV cache system prompt (B1).

        Étapes :
          1. Chargement modèle + tokenizer via mlx_load (bloquant → executor)
          2. Warm-up KV cache avec IRIS_SYSTEM_PROMPT (désactivé — voir _HAS_PROMPT_CACHE)
        """
        if self._is_loaded:
            return
        loop = asyncio.get_running_loop()
        self._model, self._processor = await loop.run_in_executor(
            None, mlx_load, self.model_path
        )
        # Extraire le tokenizer sous-jacent pour apply_chat_template
        self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
        self._is_loaded = True
        logger.info("✅ Modèle chargé : %s", self.model_name)

        if _HAS_PROMPT_CACHE:
            await loop.run_in_executor(None, self._warm_system_cache_sync)
        else:
            logger.info("ℹ️  KV Cache B1 : désactivé (mlx-vlm — re-implémentation à venir)")

    def _warm_system_cache_sync(self) -> None:
        """
        Pré-remplit le KV cache avec le system prompt d'Iris.
        Synchrone — appelé via run_in_executor au démarrage.

        Résultat :
          self._prompt_cache     → KVCache avec tokens system pré-calculés
          self._system_cache_len → nombre de tokens dans le baseline

        Gestion mémoire :
          Sur 48 Go, le modèle Q8_0 occupe ~28 Go. Le cache system prompt
          est négligeable (~quelques Mo). Safe dans l'enveloppe 48 Go.
        """
        try:
            formatted = self._tokenizer.apply_chat_template(
                [{"role": "system", "content": self.IRIS_SYSTEM_PROMPT}],
                tokenize=False,
                add_generation_prompt=False,
            )
            tokens = self._tokenizer.encode(formatted)
            self._system_cache_len = len(tokens)

            self._prompt_cache = make_prompt_cache(self._model)

            # Préfill : max_tokens=1 pour déclencher le forward pass puis on stoppe
            for _ in stream_generate(
                self._model,
                self._processor,
                prompt=formatted,
                max_tokens=1,
                temp=0.0,
            ):
                break  # Un seul forward pass suffit pour remplir le cache

            # trim_prompt_cache(cache, n) SUPPRIME les n derniers tokens.
            # Après le prefill + 1 token généré : offset = system_cache_len + 1.
            # On trim 1 pour revenir au baseline system_cache_len.
            trim_prompt_cache(self._prompt_cache, 1)

            logger.info(
                "🔥 KV Cache B1 actif : %d tokens system prompt pré-remplis",
                self._system_cache_len,
            )
        except Exception as exc:
            logger.warning("⚠️  KV Cache B1 init échouée — dégradation gracieuse : %s", exc)
            self._prompt_cache = None
            self._system_cache_len = 0

    # ── Génération depuis un historique de messages ────────────────────────────

    def stream_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """
        Génère le texte chunk par chunk depuis un historique de messages.

        Args:
            messages : [{role, content}] dans l'ordre chronologique.
                       apply_chat_template() applique le template natif du modèle
                       → support multi-turn et system prompt correct.

        Thread-safe : _cache_lock sérialise l'accès au cache partagé.
        """
        if not self._is_loaded:
            raise RuntimeError("Modèle non chargé. Appelez await engine.load() d'abord.")

        formatted = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        yield from self._stream_raw(formatted, max_tokens, temperature)

    def generate_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Retourne le texte complet depuis un historique de messages."""
        return "".join(self.stream_messages(messages, max_tokens, temperature))

    # ── API compat (prompt texte brut) ────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """Génère depuis un prompt texte brut — message user unique."""
        return self.stream_messages(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """Retourne le texte complet depuis un prompt brut."""
        return "".join(self.stream(prompt, max_tokens, temperature))

    # ── Core streaming avec gestion KV Cache ──────────────────────────────────

    def _stream_raw(
        self,
        formatted_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        """
        Streaming bas niveau avec gestion du KV Cache B1.

        Avec cache actif :
          1. Acquire _cache_lock (sérialise les requêtes — GPU = 1 stream max)
          2. trim_prompt_cache → remet le cache au baseline system prompt
          3. stream_generate avec prompt_cache (désactivé pour mlx-vlm)
          4. Release lock dans finally (même si le générateur est abandonné)

        Sans cache : génération standard sans verrou.

        Logging : tokens générés et vitesse (tok/s) à la fin de chaque génération.
        """
        last = None

        if _HAS_PROMPT_CACHE and self._prompt_cache is not None:
            self._cache_lock.acquire()
            try:
                # Calcule combien de tokens ont été ajoutés depuis le baseline
                # et trim exactement ce surplus pour revenir au system prompt seul.
                current_offset = self._prompt_cache[0].offset
                surplus = current_offset - self._system_cache_len
                if surplus > 0:
                    trim_prompt_cache(self._prompt_cache, surplus)
                for response in stream_generate(
                    self._model,
                    self._processor,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                ):
                    yield response.text
                    last = response
            finally:
                self._cache_lock.release()
        else:
            for response in stream_generate(
                self._model,
                self._processor,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temp=temperature,
            ):
                yield response.text
                last = response

        if last:
            logger.info(
                "iris ▸ %d tokens @ %.1f tok/s%s",
                last.generation_tokens,
                last.generation_tps,
                " [cache B1]" if (self._prompt_cache is not None) else "",
            )

    # ── Tool Calling ──────────────────────────────────────────────────────────

    async def stream_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        registry: Any,                   # ToolRegistry — évite import circulaire
        *,
        max_tokens: int = 512,
        temperature: float = 0.15,       # Bas → tool calls plus déterministes
        max_iterations: int = 5,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Boucle tool calling avec streaming d'events typés.

        À chaque itération :
          1. Génération complète (asyncio.to_thread — non-bloquant pour l'event loop)
          2. Parsing : texte pur → final | JSON tool calls → exécution
          3. Résultats injectés dans le contexte → itération suivante

        Args:
            messages       : Historique [{role, content}] incluant le message user courant.
            tools          : List[dict] au format OpenAI (depuis registry.get_tools_for_ministral()).
            registry       : ToolRegistry pour lookup + exécution des outils.
            max_tokens     : Tokens max par génération.
            temperature    : Température (faible → déterministe pour tool calls).
            max_iterations : Hard limit boucle (protection contre loops infinies).

        Yields:
            {"type": "text_chunk",            "content": str}
            {"type": "tool_call",             "name": str,  "arguments": dict, "call_id": str}
            {"type": "tool_result",           "name": str,  "result": Any,     "call_id": str}
            {"type": "confirmation_required", "name": str,  "arguments": dict, "call_id": str}
            {"type": "final",                 "content": str}
            {"type": "error",                 "content": str}
        """
        if not self._is_loaded:
            yield {"type": "error", "content": "Modèle non chargé."}
            return

        context = list(messages)  # Copie — ne pas muter l'historique de l'appelant

        for iteration in range(max_iterations):
            logger.debug("stream_with_tools: iteration %d/%d", iteration + 1, max_iterations)

            # ── Génération (bloquante dans un thread) ─────────────────────────
            try:
                raw_response = await asyncio.to_thread(
                    self._generate_raw,
                    context,
                    tools,
                    max_tokens,
                    temperature,
                )
            except Exception as exc:
                logger.error("Erreur génération iter %d : %s", iteration + 1, exc)
                yield {"type": "error", "content": f"Erreur génération : {exc}"}
                return

            logger.debug("Réponse brute [%d chars]: %r", len(raw_response), raw_response[:200])

            # ── Parsing ───────────────────────────────────────────────────────
            tool_calls, text_content = self._parse_tool_calls(raw_response)

            if not tool_calls:
                if text_content:
                    yield {"type": "text_chunk", "content": text_content}
                yield {"type": "final", "content": text_content}
                return

            # ── Exécution des tool calls ───────────────────────────────────────
            assistant_msg = self._build_assistant_message(text_content, tool_calls)
            context.append(assistant_msg)

            for tc in tool_calls:
                call_id = tc["id"]
                name = tc["name"]
                arguments = tc["arguments"]

                yield {"type": "tool_call", "name": name, "arguments": arguments, "call_id": call_id}

                schema = registry.get(name)
                if schema and schema.requires_confirmation:
                    yield {
                        "type": "confirmation_required",
                        "name": name,
                        "arguments": arguments,
                        "call_id": call_id,
                    }
                    context.append({
                        "role": "tool",
                        "content": f"Tool '{name}' requires user confirmation before execution.",
                        "tool_call_id": call_id,
                    })
                    continue

                result = await self._execute_tool(tc, registry)
                yield {"type": "tool_result", "name": name, "result": result, "call_id": call_id}

                context.append({
                    "role": "tool",
                    "content": self._result_to_string(result),
                    "tool_call_id": call_id,
                })

        # Hard limit atteinte
        logger.warning("stream_with_tools: max_iterations=%d atteint", max_iterations)
        yield {
            "type": "error",
            "content": f"Limite d'itérations atteinte ({max_iterations}). Arrêt.",
        }
        yield {"type": "final", "content": ""}

    # ── Helpers privés : génération ───────────────────────────────────────────

    def _generate_raw(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Génération synchrone complète (pour le tool calling loop).
        Appelée via asyncio.to_thread().

        Tente d'injecter les tools via apply_chat_template(tools=...).
        Si le tokenizer ne supporte pas ce paramètre, fallback vers injection
        manuelle dans le system prompt.
        """
        try:
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tools=tools if tools else None,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            logger.warning("Tokenizer ne supporte pas tools=, fallback injection system prompt")
            messages_with_tools = self._inject_tools_system_prompt(messages, tools or [])
            formatted = self._tokenizer.apply_chat_template(
                messages_with_tools,
                tokenize=False,
                add_generation_prompt=True,
            )

        return "".join(self._stream_raw(formatted, max_tokens, temperature))

    @staticmethod
    def _inject_tools_system_prompt(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Fallback : injecte les tools dans un message system en tête de conversation.
        Utilisé si apply_chat_template() ne supporte pas le paramètre tools=.
        """
        tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
        system_content = (
            "You have access to the following tools. "
            "To call a tool, respond with JSON: "
            '{"tool_calls": [{"name": "tool_name", "arguments": {...}}]}\n\n'
            f"Available tools:\n{tools_json}"
        )

        result = list(messages)
        if result and result[0].get("role") == "system":
            result[0] = {
                "role": "system",
                "content": result[0]["content"] + "\n\n" + system_content,
            }
        else:
            result.insert(0, {"role": "system", "content": system_content})

        return result

    # ── Helpers privés : parsing ──────────────────────────────────────────────

    @staticmethod
    def _extract_json_object(text: str, start: int) -> tuple[dict | None, int]:
        """
        Extrait un objet JSON débutant à `start` en comptant les accolades.
        Retourne (dict_parsé, position_fin) ou (None, start) si échec.
        Plus robuste qu'une regex : gère les accolades imbriquées dans les valeurs.
        """
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj, i + 1
                    except json.JSONDecodeError:
                        return None, start

        return None, start

    @staticmethod
    def _extract_json_array(text: str, start: int) -> tuple[list | None, int]:
        """Extrait un tableau JSON débutant à `start` en comptant les crochets."""
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        arr = json.loads(text[start : i + 1])
                        return arr, i + 1
                    except json.JSONDecodeError:
                        return None, start

        return None, start

    @classmethod
    def _parse_tool_calls(cls, response: str) -> tuple[list[dict], str]:
        """
        Extrait les tool calls d'une réponse brute du modèle.
        Retourne (tool_calls, texte_restant).

        Formats gérés :
          1. [TOOL_CALLS] [...] — token Mistral natif (Tekken tokenizer)
          2. {"tool_calls": [...]} — format JSON objet
        """
        response = response.strip()

        # Format 1 : [TOOL_CALLS] array (Mistral Tekken natif)
        if _TOOL_CALLS_TOKEN in response:
            token_idx = response.index(_TOOL_CALLS_TOKEN)
            after_token = response[token_idx + len(_TOOL_CALLS_TOKEN):].lstrip()

            if after_token.startswith("["):
                arr, end = cls._extract_json_array(after_token, 0)
                if arr is not None:
                    calls = cls._normalize_tool_calls(arr)
                    if calls:
                        text_before = response[:token_idx].strip()
                        text_after = after_token[end:].strip()
                        remaining = " ".join(filter(None, [text_before, text_after]))
                        logger.debug("Parsed %d tool call(s) via [TOOL_CALLS] token", len(calls))
                        return calls, remaining

        # Format 2 : JSON objet contenant "tool_calls"
        for i, ch in enumerate(response):
            if ch == "{":
                obj, end = cls._extract_json_object(response, i)
                if obj is not None and "tool_calls" in obj:
                    calls = cls._normalize_tool_calls(obj["tool_calls"])
                    if calls:
                        text_before = response[:i].strip()
                        text_after = response[end:].strip()
                        remaining = " ".join(filter(None, [text_before, text_after]))
                        logger.debug("Parsed %d tool call(s) via JSON object", len(calls))
                        return calls, remaining

        return [], response

    @staticmethod
    def _normalize_tool_calls(raw: list[Any]) -> list[dict[str, Any]]:
        """
        Normalise une liste de tool calls vers : {"id": str, "name": str, "arguments": dict}

        Formats supportés :
          - Mistral natif  : {"name": "...", "arguments": {...}}
          - OpenAI-like    : {"type": "function", "id": "...", "function": {"name": "...", "arguments": "..."}}
        """
        normalized: list[dict[str, Any]] = []

        for i, call in enumerate(raw):
            if not isinstance(call, dict):
                continue

            call_id = call.get("id", f"call_{i}")

            if "name" in call:
                name = call["name"]
                args = call.get("arguments", {})
            elif "function" in call:
                func = call["function"]
                name = func.get("name", "")
                args = func.get("arguments", {})
                call_id = call.get("id", call_id)
            else:
                logger.warning("Tool call format inconnu (index %d): %s", i, call)
                continue

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning("arguments JSON invalide pour '%s': %r", name, args)
                    args = {"raw_arguments": args}

            if not isinstance(args, dict):
                args = {}

            if name:
                normalized.append({"id": call_id, "name": name, "arguments": args})

        return normalized

    # ── Helpers privés : contexte et exécution ────────────────────────────────

    @staticmethod
    def _build_assistant_message(
        text: str,
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Construit le message assistant avec tool_calls pour l'historique contexte.
        Format requis par apply_chat_template pour l'itération suivante.
        """
        return {
            "role": "assistant",
            "content": text or "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                    },
                }
                for tc in tool_calls
            ],
        }

    @staticmethod
    async def _execute_tool(
        tool_call: dict[str, Any],
        registry: Any,
    ) -> Any:
        """
        Exécute un outil et retourne son résultat.

        Gestion d'erreurs :
          - Outil introuvable → dict erreur avec liste des outils disponibles
          - Exception pendant exécution → dict erreur (jamais crash)
        """
        name = tool_call["name"]
        arguments = tool_call.get("arguments", {})

        executor = registry.get_executor(name)
        if executor is None:
            available = registry.names
            logger.warning("Outil '%s' introuvable. Disponibles : %s", name, available)
            return {
                "error": f"Tool '{name}' not found.",
                "available_tools": available,
                "success": False,
            }

        try:
            if inspect.iscoroutinefunction(executor):
                result = await executor(**arguments)
            else:
                result = await asyncio.to_thread(executor, **arguments)
            return result
        except TypeError as exc:
            logger.error("Outil '%s' : arguments invalides (%s) → %s", name, arguments, exc)
            return {
                "error": f"Invalid arguments for '{name}': {exc}",
                "received_arguments": arguments,
                "success": False,
            }
        except Exception as exc:
            logger.error("Outil '%s' : exception inattendue : %s", name, exc)
            return {
                "error": f"Tool '{name}' failed: {exc}",
                "success": False,
            }

    @staticmethod
    def _result_to_string(result: Any) -> str:
        """Convertit un résultat d'outil en string pour l'injection dans le contexte LLM."""
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)
