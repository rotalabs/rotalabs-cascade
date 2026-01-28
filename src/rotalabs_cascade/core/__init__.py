"""Core module for rotalabs-cascade orchestration engine.

This module provides the foundational components for configuration-driven
multi-stage decision routing and execution.
"""

from rotalabs_cascade.core.config import (
    CascadeConfig,
    Condition,
    ConditionOperator,
    RoutingAction,
    RoutingRule,
    StageConfig,
)
from rotalabs_cascade.core.context import ExecutionContext, StageResult
from rotalabs_cascade.core.engine import CascadeEngine

__all__ = [
    "CascadeEngine",
    "CascadeConfig",
    "StageConfig",
    "RoutingRule",
    "Condition",
    "RoutingAction",
    "ConditionOperator",
    "ExecutionContext",
    "StageResult",
]
