"""
coding/graph.py — Agent de coding sandboxé.

Modèle d'accès :
  - Lecture  : libre sur tout le système (read_file, ls, glob, grep)
  - Écriture : bloquée hors sandbox au niveau backend (refus net)
  - Shell    : HITL obligatoire avant chaque commande

Sandbox filesystem : /Volumes/AISSD/iris-sandbox
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ShellToolMiddleware,
    TodoListMiddleware,
)
from deepagents.middleware import FilesystemMiddleware
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import WriteResult, EditResult
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
SANDBOX_ROOT = Path("/Volumes/AISSD/iris-sandbox")

# ---------------------------------------------------------------------------
# Backend : lecture libre, écriture confinée au sandbox
# ---------------------------------------------------------------------------
class SandboxedWriteBackend(FilesystemBackend):
    """
    Lecture : accès libre à tout le système réel.
    Écriture / édition : bloquée hors SANDBOX_ROOT (refus net, pas de HITL).

    Le répertoire de travail par défaut est SANDBOX_ROOT, mais l'agent
    peut passer des chemins absolus pour lire n'importe quel fichier.
    """

    def _assert_write_allowed(self, path: str) -> None:
        real = Path(path).resolve()
        sandbox = SANDBOX_ROOT.resolve()
        if not str(real).startswith(str(sandbox)):
            raise PermissionError(
                f"Écriture hors sandbox refusée : {path}\n"
                f"→ Copie d'abord le fichier dans {SANDBOX_ROOT}/workspace/ "
                f"avant de le modifier."
            )

    def write(self, path: str, content: str, **kwargs) -> WriteResult:
        self._assert_write_allowed(path)
        return super().write(path, content, **kwargs)

    def edit(self, path: str, **kwargs) -> EditResult:
        self._assert_write_allowed(path)
        return super().edit(path, **kwargs)


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
model = ChatAnthropic(
    model=os.environ["OMLX_MODEL"],
    thinking={"type": "enabled", "budget_tokens": 8192},
    max_tokens=16384,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
)

# ---------------------------------------------------------------------------
# Graph (async context manager — requis par MultiServerMCPClient)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def make_graph():
    async with MultiServerMCPClient({
        "docs-langchain": {
            "transport": "http",
            "url": "https://docs.langchain.com/mcp",
        }
    }) as mcp_client:
        doc_tools = await mcp_client.get_tools()

        coding_agent = create_agent(
            model=model,
            tools=doc_tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=[
                # Planification des tâches multi-étapes
                TodoListMiddleware(),

                # Filesystem :
                #   - root_dir = sandbox (répertoire de travail par défaut)
                #   - virtual_mode = False → les chemins absolus fonctionnent
                #     pour la lecture (pas de restriction)
                #   - SandboxedWriteBackend bloque les écritures hors sandbox
                FilesystemMiddleware(
                    backend=SandboxedWriteBackend(
                        root_dir=str(SANDBOX_ROOT),
                        virtual_mode=False,
                    ),
                ),

                # Shell persistant — toutes les commandes requièrent une approbation
                ShellToolMiddleware(),

                # HITL uniquement sur le shell
                # (les écritures hors sandbox sont bloquées directement
                # par le backend, pas besoin de HITL dessus)
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "shell": True,
                    },
                    description_prefix="Commande shell en attente de validation",
                ),
            ],
            checkpointer=InMemorySaver(),
        )

        yield coding_agent


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""\
You are an expert software engineer and coding assistant integrated into the Iris \
assistant system.

## Filesystem access model
- **Read** : you can read any file on the system using its absolute path.
- **Write / edit** : only allowed inside your sandbox ({SANDBOX_ROOT}).
  Any attempt to write outside the sandbox will be refused immediately.
- **Shell** : every command requires human approval before execution.

## Typical workflow
1. Use write_todos to plan complex, multi-step tasks.
2. Explore the real codebase freely with read_file and ls using absolute paths.
3. Copy the files you need to modify into {SANDBOX_ROOT}/workspace/ first.
4. Work on the copies — read, edit, test.
5. When you need to run code, propose the shell command and wait for approval.
6. Once validated, propose copying the result back to the original location \
via a shell command (which will also require approval).

## Guidelines
- Prefer small, incremental edits over full rewrites.
- Always re-read a file after editing it to verify the change.
- Explain the impact of any change before proposing it.
- Use the LangChain docs tool to look up APIs when needed.
- Never store secrets or API keys in the sandbox.
"""