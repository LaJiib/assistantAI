from datetime import datetime

import pytz
from langchain.agents.middleware import AgentMiddleware, dynamic_prompt
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deep_research.configuration import Configuration
from deep_research.prompts import RESEARCHER_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT
from deep_research.state import ResearchState


class UrlTrackingMiddleware(AgentMiddleware[ResearchState]):
    """Intercepts fetch_webpage calls to append the fetched URL to state."""

    state_schema = ResearchState

    def _track(self, result, tool_name: str, tool_args: dict):
        if tool_name == "fetch_webpage":
            url = (tool_args or {}).get("url", "")
            if url and isinstance(result, ToolMessage):
                return Command(update={"url_fetched": [url], "messages": [result]})
        return result

    def wrap_tool_call(self, request, handler):
        return self._track(
            handler(request),
            request.tool_call.get("name", ""),
            request.tool_call.get("args") or {},
        )

    async def awrap_tool_call(self, request, handler):
        return self._track(
            await handler(request),
            request.tool_call.get("name", ""),
            request.tool_call.get("args") or {},
        )


def _today() -> str:
    return datetime.now(pytz.timezone("Europe/Paris")).strftime("%A, %B %d, %Y")


def _url_section(state: dict) -> str:
    urls = sorted(set(state.get("url_fetched", [])))
    return "\n".join(f"- {u}" for u in urls) if urls else "None yet."


@dynamic_prompt
def researcher_dynamic_prompt(request):
    cfg = Configuration()
    return (
        f"{RESEARCHER_SYSTEM_PROMPT}\n\n"
        f"Today: {_today()}\n"
        f"Max tool calls: {cfg.max_react_tool_calls}\n\n"
        f"URLs already fetched in this session (avoid re-fetching):\n"
        f"{_url_section(request.state)}"
    )


@dynamic_prompt
def supervisor_dynamic_prompt(request):
    cfg = Configuration()
    call_count = request.state.get("researcher_call_count", 0)
    return (
        f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
        f"Today: {_today()}\n\n"
        f"## Current situational context\n"
        f"- Researcher calls so far: {call_count} / {cfg.max_researcher_iterations} (max total)\n"
        f"- Max parallel workers per batch: {cfg.max_concurrent_research_units}\n\n"
        f"URLs fetched by researcher agents so far:\n"
        f"{_url_section(request.state)}"
    )
