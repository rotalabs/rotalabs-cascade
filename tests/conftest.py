"""Pytest fixtures for rotalabs-cascade tests.

This module provides reusable fixtures for testing the cascade orchestration engine.
"""

import pytest
from typing import Any, Dict

from rotalabs_cascade.core.config import (
    CascadeConfig,
    Condition,
    ConditionOperator,
    RoutingAction,
    RoutingRule,
    StageConfig,
)
from rotalabs_cascade.core.context import ExecutionContext


@pytest.fixture
def simple_config() -> CascadeConfig:
    """Create a simple cascade configuration with 3 stages.

    Stages:
        - FAST: Quick initial processing
        - MEDIUM: Moderate complexity processing
        - SLOW: Deep analysis (initially disabled)
    """
    stages = {
        "FAST": StageConfig(
            name="FAST",
            enabled=True,
            timeout_ms=1000,
            routing_rules=[
                RoutingRule(
                    name="enable_medium_if_low_confidence",
                    type="routing",
                    condition=Condition(
                        field="confidence",
                        operator=ConditionOperator.LT,
                        value=0.8,
                    ),
                    action=RoutingAction(
                        type="enable_stages",
                        stages=["MEDIUM"],
                    ),
                    priority=10,
                )
            ],
        ),
        "MEDIUM": StageConfig(
            name="MEDIUM",
            enabled=False,
            timeout_ms=5000,
            depends_on=["FAST"],
            routing_rules=[
                RoutingRule(
                    name="enable_slow_if_uncertain",
                    type="routing",
                    condition=Condition(
                        field="confidence",
                        operator=ConditionOperator.LT,
                        value=0.5,
                    ),
                    action=RoutingAction(
                        type="enable_stages",
                        stages=["SLOW"],
                    ),
                    priority=10,
                )
            ],
        ),
        "SLOW": StageConfig(
            name="SLOW",
            enabled=False,
            timeout_ms=10000,
            depends_on=["MEDIUM"],
        ),
    }

    return CascadeConfig(
        name="simple_cascade",
        version="1.0.0",
        stages=stages,
        execution_order=["FAST", "MEDIUM", "SLOW"],
        global_timeout_ms=30000,
        max_parallel_stages=3,
    )


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Create sample input data for cascade execution.

    Returns:
        Dictionary with typical input structure including user info,
        request details, and configuration.
    """
    return {
        "user_id": "user123",
        "request": {
            "text": "Sample request text",
            "priority": "high",
            "metadata": {
                "source": "api",
                "timestamp": 1234567890,
            },
        },
        "config": {
            "threshold": 0.7,
            "max_results": 10,
        },
        "confidence": 0.9,
    }


@pytest.fixture
def async_handler():
    """Factory fixture for creating async stage handlers.

    Returns a function that creates async handlers with customizable behavior.

    Example:
        handler = async_handler(result="success", confidence=0.95)
        result = await handler(context)
    """
    def _create_handler(
        result: Any = "success",
        confidence: float = 0.9,
        delay_ms: int = 0,
        should_fail: bool = False,
        error_message: str = "Handler error",
        data: Dict[str, Any] = None,
    ):
        """Create an async handler with specified behavior.

        Args:
            result: Result value to return
            confidence: Confidence score (0-1)
            delay_ms: Simulated processing delay in milliseconds
            should_fail: Whether handler should raise an exception
            error_message: Error message if should_fail=True
            data: Additional data to include in result

        Returns:
            Async handler function
        """
        async def handler(context: ExecutionContext) -> Dict[str, Any]:
            import asyncio

            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)

            if should_fail:
                raise RuntimeError(error_message)

            return {
                "result": result,
                "confidence": confidence,
                "data": data or {},
            }

        return handler

    return _create_handler


@pytest.fixture
def execution_context(sample_data) -> ExecutionContext:
    """Create an execution context initialized with sample data.

    Args:
        sample_data: Sample data fixture

    Returns:
        Initialized ExecutionContext
    """
    return ExecutionContext(sample_data)


@pytest.fixture
def complex_condition() -> Condition:
    """Create a complex composite condition for testing.

    Returns a condition with nested AND/OR logic.
    """
    return Condition(
        operator=ConditionOperator.AND,
        conditions=[
            Condition(
                field="confidence",
                operator=ConditionOperator.GT,
                value=0.5,
            ),
            Condition(
                operator=ConditionOperator.OR,
                conditions=[
                    Condition(
                        field="request.priority",
                        operator=ConditionOperator.EQ,
                        value="high",
                    ),
                    Condition(
                        field="user_id",
                        operator=ConditionOperator.IN,
                        value=["user123", "user456"],
                    ),
                ],
            ),
        ],
    )
