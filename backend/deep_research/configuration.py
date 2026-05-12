"""Configuration for the Deep Research subagent."""

import os
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):

    # Limites — réduites pour modèle local
    max_structured_output_retries: int = Field(default=3)
    max_concurrent_research_units: int = Field(
        default=2,
        description="Workers parallèles max. Limité à 2 pour le KV cache local."
    )
    max_researcher_iterations: int = Field(
        default=3,
        description="Iterations du superviseur. Réduit de 6 à 3 pour le local."
    )
    max_react_tool_calls: int = Field(
        default=6,
        description="Appels tools max par worker. Réduit de 10 à 6."
    )

    # Limites de tokens par rôle
    research_model_max_tokens: int = Field(default=4096)
    compression_model_max_tokens: int = Field(default=4096)
    final_report_model_max_tokens: int = Field(default=6144)

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        arbitrary_types_allowed = True