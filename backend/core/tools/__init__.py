# core/tools/__init__.py
"""
Tool calling infrastructure — schémas Pydantic v2 conformes au standard OpenAI.
"""

from .registry import SystemToolProtectedError, ToolRegistry
from .schema import JsonSchemaType, PermissionLevel, ToolParameter, ToolSchema

__all__ = [
    "JsonSchemaType",
    "PermissionLevel",
    "SystemToolProtectedError",
    "ToolParameter",
    "ToolRegistry",
    "ToolSchema",
]
