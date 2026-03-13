"""
MinistralEngine — wrapper mlx-lm pour Ministral 3 14B Instruct.

Cycle de vie :
    engine = MinistralEngine(model_path)
    await engine.load()                       # startup FastAPI
    engine.stream(prompt, ...)                # sync, itère des chunks texte (compat Phase 2)
    engine.generate(prompt, ...)              # sync, retourne le texte complet (compat Phase 2)
    engine.stream_messages(messages, ...)     # sync, multi-turn via historique messages
    engine.generate_messages(messages, ...)   # sync, multi-turn, retourne texte complet
    engine.stream_with_tools(...)             # async, tool calling loop avec events typés

Tool calling (stream_with_tools) :
    Events yielded : dict avec champ "type" :
        {"type": "text_chunk",           "content": str}
        {"type": "tool_call",            "name": str, "arguments": dict, "call_id": str}
        {"type": "tool_result",          "name": str, "result": Any,    "call_id": str}
        {"type": "confirmation_required","name": str, "arguments": dict, "call_id": str}
        {"type": "final",                "content": str}
        {"type": "error",                "content": str}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Iterator, List

from mlx_lm import load as mlx_load, stream_generate
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger(__name__)

# Token spécial Mistral (Tekken tokenizer) signalant un tool call dans la réponse
_TOOL_CALLS_TOKEN = "[TOOL_CALLS]"


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

    # ── Tool Calling ──────────────────────────────────────────────────────────

    async def stream_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        registry: Any,                   # ToolRegistry — import évite circular dep
        *,
        max_tokens: int = 512,
        temperature: float = 0.15,       # Plus bas que génération texte → tool calls précis
        max_iterations: int = 5,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Boucle tool calling avec streaming d'events typés.

        À chaque itération :
          1. Génération complète (via asyncio.to_thread — non-bloquant pour l'event loop)
          2. Parsing : texte pur → final | JSON tool calls → exécution
          3. Résultats injectés dans le contexte → itération suivante

        Args:
            messages:       Historique [{role, content}] incluant le message user courant.
            tools:          List[dict] au format OpenAI (depuis registry.get_tools_for_ministral()).
            registry:       ToolRegistry pour lookup + exécution des outils.
            max_tokens:     Tokens max par génération.
            temperature:    Température (faible → déterministe pour tool calls).
            max_iterations: Hard limit boucle (défaut 5, protection contre loops infinies).

        Yields:
            {"type": "text_chunk",            "content": str}
            {"type": "tool_call",             "name": str,  "arguments": dict, "call_id": str}
            {"type": "tool_result",           "name": str,  "result": Any,     "call_id": str}
            {"type": "confirmation_required", "name": str,  "arguments": dict, "call_id": str}
            {"type": "final",                 "content": str}
            {"type": "error",                 "content": str}

        Backward compat : stream(), generate(), stream_messages() sont inchangés.
        """
        if not self._is_loaded:
            yield {"type": "error", "content": "Modèle non chargé."}
            return

        context = list(messages)  # copie — ne pas muter l'historique de l'appelant

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
                # Réponse finale — pas de tool call détecté
                if text_content:
                    yield {"type": "text_chunk", "content": text_content}
                yield {"type": "final", "content": text_content}
                return

            # ── Exécution des tool calls ───────────────────────────────────────
            # Construire le message assistant avec les tool_calls
            assistant_msg = self._build_assistant_message(text_content, tool_calls)
            context.append(assistant_msg)

            for tc in tool_calls:
                call_id = tc["id"]
                name = tc["name"]
                arguments = tc["arguments"]

                yield {"type": "tool_call", "name": name, "arguments": arguments, "call_id": call_id}

                # Vérification confirmation requise
                schema = registry.get(name)
                if schema and schema.requires_confirmation:
                    yield {
                        "type": "confirmation_required",
                        "name": name,
                        "arguments": arguments,
                        "call_id": call_id,
                    }
                    # Inject avertissement dans contexte — modèle informe l'utilisateur
                    context.append({
                        "role": "tool",
                        "content": f"Tool '{name}' requires user confirmation before execution.",
                        "tool_call_id": call_id,
                    })
                    continue

                # Exécution
                result = await self._execute_tool(tc, registry)
                yield {"type": "tool_result", "name": name, "result": result, "call_id": call_id}

                # Résultat injecté dans le contexte pour l'itération suivante
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
        Génération synchrone complète. Appelée via asyncio.to_thread().

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
            # Tokenizer ne supporte pas tools= → fallback system prompt
            logger.warning(
                "Tokenizer ne supporte pas tools=, fallback injection system prompt"
            )
            messages_with_tools = self._inject_tools_system_prompt(messages, tools or [])
            formatted = self._tokenizer.apply_chat_template(
                messages_with_tools,
                tokenize=False,
                add_generation_prompt=True,
            )

        chunks: list[str] = []
        last = None
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        ):
            chunks.append(response.text)
            last = response

        if last:
            logger.info(
                "tool_iter: %d tokens @ %.1f tok/s",
                last.generation_tokens,
                last.generation_tps,
            )

        return "".join(chunks)

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
            # Ajouter aux instructions système existantes
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
        """
        Extrait un tableau JSON débutant à `start` en comptant les crochets.
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
        tool_calls = [] si aucun tool call détecté.

        Formats gérés :
        1. [TOOL_CALLS] [...] — token Mistral natif (Tekken tokenizer)
        2. {"tool_calls": [...]} — format JSON objet
        3. Texte mixte contenant l'un des formats ci-dessus

        Chaque tool call normalisé : {"id": str, "name": str, "arguments": dict}
        """
        response = response.strip()

        # ── Format 1 : [TOOL_CALLS] array (Mistral Tekken natif) ──────────────
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

        # ── Format 2 : JSON objet contenant "tool_calls" ──────────────────────
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

        # ── Pas de tool call détecté ───────────────────────────────────────────
        return [], response

    @staticmethod
    def _normalize_tool_calls(raw: list[Any]) -> list[dict[str, Any]]:
        """
        Normalise une liste de tool calls depuis n'importe quel format vers :
        {"id": str, "name": str, "arguments": dict}

        Formats supportés en entrée :
        - Mistral natif  : {"name": "...", "arguments": {...}}
        - OpenAI-like    : {"type": "function", "id": "...", "function": {"name": "...", "arguments": "..."}}
        """
        normalized: list[dict[str, Any]] = []

        for i, call in enumerate(raw):
            if not isinstance(call, dict):
                continue

            call_id = call.get("id", f"call_{i}")

            # Format Mistral natif
            if "name" in call:
                name = call["name"]
                args = call.get("arguments", {})

            # Format OpenAI (function wrapper)
            elif "function" in call:
                func = call["function"]
                name = func.get("name", "")
                args = func.get("arguments", {})
                call_id = call.get("id", call_id)

            else:
                logger.warning("Tool call format inconnu (index %d): %s", i, call)
                continue

            # arguments peut être une string JSON (format OpenAI) ou un dict
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

        Le résultat brut (str, dict, etc.) est retourné tel quel pour l'event
        "tool_result". Il sera stringifié pour l'injection dans le contexte LLM.
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
            if asyncio.iscoroutinefunction(executor):
                result = await executor(**arguments)
            else:
                result = await asyncio.to_thread(executor, **arguments)
            return result
        except TypeError as exc:
            # Arguments incorrects (type mismatch ou paramètre manquant)
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
        """
        Convertit un résultat d'outil en string pour l'injection dans le contexte LLM.
        Le LLM lit ce string comme contenu du message "tool".
        """
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)
