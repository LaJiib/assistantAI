from dotenv import load_dotenv
import os
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool, InjectedToolCallId, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from utils.tools import web_search, fetch_webpage
from deep_research.middleware import (
    UrlTrackingMiddleware,
    researcher_dynamic_prompt,
    supervisor_dynamic_prompt,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Models — separate instances for researcher and supervisor token budgets
# ---------------------------------------------------------------------------
_base = dict(
    model=os.environ["OMLX_MODEL"],
    base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "local"),
)
additional_kwargs_researcher = {"chat_template_kwargs": {"enable_thinking": True}, "thinking_tokens": 8192, "max_tokens": 16384}
researcher_model = ChatOpenAI(
    **_base,
    model_kwargs= {
        "extra_body": additional_kwargs_researcher,
    },
)

additional_kwargs_supervisor = {"chat_template_kwargs": {"enable_thinking": True}, "thinking_tokens": 16384, "max_tokens": 32768}
supervisor_model = ChatOpenAI(
    **_base,
    model_kwargs= {
        "extra_body": additional_kwargs_supervisor,
    },
)

# ---------------------------------------------------------------------------
# Researcher — stateless, fresh context per invocation
# ---------------------------------------------------------------------------
researcher = create_agent(
    researcher_model,
    tools=[web_search, fetch_webpage],
    middleware=[
        TodoListMiddleware(),
        UrlTrackingMiddleware(),
        researcher_dynamic_prompt,
    ],
)

# ---------------------------------------------------------------------------
# Researcher tool — wraps researcher.ainvoke and propagates url_fetched
# ---------------------------------------------------------------------------
@tool(
    "researcher",
    description=(
        "Searches the web and reads pages to investigate a specific research axis. "
        "Provide a clear directive describing exactly what to investigate and what "
        "information to return. Returns a detailed synthesis with key findings and "
        "source URLs."
    ),
)
async def call_researcher(
    directive: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    runtime: ToolRuntime,
) -> Command:
    call_n = runtime.state.get("researcher_call_count", 0) + 1
    result = await researcher.ainvoke(
        {"messages": [{"role": "user", "content": directive}]},
        config={"run_name": f"researcher_{call_n}"},
    )
    return Command(
        update={
            "url_fetched": result.get("url_fetched", []),
            "researcher_call_count": 1,
            "messages": [
                ToolMessage(
                    content=result["messages"][-1].content,
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )

# ---------------------------------------------------------------------------
# Supervisor — the deep_research entry point exposed to Iris
# ---------------------------------------------------------------------------
deep_researcher = create_agent(
    supervisor_model,
    tools=[call_researcher],
    middleware=[
        UrlTrackingMiddleware(),
        supervisor_dynamic_prompt,
    ],
)
