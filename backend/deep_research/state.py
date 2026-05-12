import operator
from typing import Annotated

from langchain.agents.middleware import AgentState


class ResearchState(AgentState):
    url_fetched: Annotated[list[str], operator.add]
    researcher_call_count: Annotated[int, operator.add]
