"""Execution context for cascade orchestration.

This module provides the execution context that tracks state, results, and
metadata throughout the cascade execution lifecycle.

Supports both:
- Flat dictionary input: {"user_id": "123", "amount": 100}
- EventWithContext input: EventWithContext(event=..., context=...)
"""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from rotalabs_cascade.core.event import EventWithContext


class StageResult:
    """Represents the result of a stage execution.

    Uses __slots__ for memory efficiency.

    Attributes:
        stage_name: Name of the executed stage.
        result: Result value from the stage handler.
        confidence: Confidence score (0-1) if applicable.
        data: Additional data returned by the stage.
        error: Error message if stage failed.
        time_ms: Execution time in milliseconds.
    """

    __slots__ = ("stage_name", "result", "confidence", "data", "error", "time_ms")

    def __init__(
        self,
        stage_name: str,
        result: Any = None,
        confidence: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        time_ms: float = 0.0,
    ):
        self.stage_name = stage_name
        self.result = result
        self.confidence = confidence
        self.data = data or {}
        self.error = error
        self.time_ms = time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert stage result to dictionary."""
        result = {
            "stage_name": self.stage_name,
            "result": self.result,
            "time_ms": self.time_ms,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.data:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        return result

    def __repr__(self) -> str:
        return f"StageResult(stage_name={self.stage_name!r}, result={self.result!r}, error={self.error!r})"


class ExecutionContext:
    """Execution context for cascade orchestration.

    Manages state, results, and metadata throughout cascade execution.
    Uses __slots__ for memory efficiency and zero-copy reference to input data.

    Attributes:
        _data: Reference to input data (zero-copy).
        _stage_results: List of stage execution results.
        _errors: List of error messages.
        _metadata: Additional metadata dictionary.
        _start_time: Execution start timestamp.
        _termination_flag: Whether execution should terminate.
        _next_stage: Next stage to execute (for skip_to actions).
        _enabled_stages: Set of dynamically enabled stage names.
        _disabled_stages: Set of dynamically disabled stage names.
        _cache: Cache for computed values (dot notation lookups).
        _execution_order: Planned execution order.
        _timeline: Timeline of execution events.
        _routing_decisions: List of routing decisions made.
    """

    __slots__ = (
        "_data",
        "_event_with_context",
        "_stage_results",
        "_errors",
        "_metadata",
        "_start_time",
        "_termination_flag",
        "_next_stage",
        "_enabled_stages",
        "_disabled_stages",
        "_cache",
        "_execution_order",
        "_timeline",
        "_routing_decisions",
    )

    def __init__(self, data: Union[Dict[str, Any], "EventWithContext"]):
        """Initialize execution context.

        Args:
            data: Input data - either a dictionary or EventWithContext object.
                  If EventWithContext, it will be converted to flat dict for access.

        Examples:
            # Flat dictionary (backward compatible)
            ctx = ExecutionContext({"user_id": "123", "amount": 100})

            # EventWithContext (domain-agnostic)
            ctx = ExecutionContext(EventWithContext(event=..., context=...))
        """
        # Handle EventWithContext input
        if hasattr(data, "to_flat_dict"):
            self._data = data.to_flat_dict()
            self._event_with_context = data
        else:
            self._data = data  # Zero-copy reference
            self._event_with_context = None
        self._stage_results: List[StageResult] = []
        self._errors: List[str] = []
        self._metadata: Dict[str, Any] = {}
        self._start_time = time.time()
        self._termination_flag = False
        self._next_stage: Optional[str] = None
        self._enabled_stages: set = set()
        self._disabled_stages: set = set()
        self._cache: Dict[str, Any] = {}
        self._execution_order: List[str] = []
        self._timeline: List[Dict[str, Any]] = []
        self._routing_decisions: List[Dict[str, Any]] = []

    @property
    def data(self) -> Dict[str, Any]:
        """Get reference to input data."""
        return self._data

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed execution time in milliseconds."""
        return (time.time() - self._start_time) * 1000

    @property
    def should_terminate(self) -> bool:
        """Check if execution should terminate."""
        return self._termination_flag

    def get(self, path: str, default: Any = None) -> Any:
        """Get value from data or stage results using dot notation with caching.

        Supports paths like:
        - "user.profile.age" - looks up in input data
        - "stages.FAST_CHECK.confidence" - looks up in stage results

        Args:
            path: Dot-separated path (e.g., "user.profile.age").
            default: Default value if path not found.

        Returns:
            Value at path or default.

        Examples:
            >>> ctx.get("user.name")
            'John'
            >>> ctx.get("stages.FAST_CHECK.confidence")
            0.85
        """
        # Check cache first
        if path in self._cache:
            return self._cache[path]

        # Navigate path
        parts = path.split(".")

        # Handle stage results path (stages.STAGE_NAME.field)
        if parts[0] == "stages" and len(parts) >= 2:
            stage_name = parts[1]
            stage_result = self.get_stage_result(stage_name)
            if stage_result is None:
                return default

            if len(parts) == 2:
                # Return entire stage result as dict
                value = stage_result.to_dict()
            else:
                # Navigate into stage result
                field = parts[2]
                if hasattr(stage_result, field):
                    value = getattr(stage_result, field)
                elif field in (stage_result.data or {}):
                    value = stage_result.data[field]
                else:
                    return default

                # Handle deeper paths in data
                if len(parts) > 3 and isinstance(value, dict):
                    for part in parts[3:]:
                        if isinstance(value, dict):
                            value = value.get(part)
                            if value is None:
                                return default
                        else:
                            return default

            self._cache[path] = value
            return value

        # Navigate input data
        value = self._data

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    return default
            else:
                return default

        # Cache result
        self._cache[path] = value
        return value

    def set(self, path: str, value: Any) -> None:
        """Set value in data using dot notation.

        Args:
            path: Dot-separated path (e.g., "user.profile.age").
            value: Value to set.

        Examples:
            >>> ctx.set("user.name", "Jane")
            >>> ctx.set("user.settings.theme", "dark")
        """
        parts = path.split(".")
        target = self._data

        # Navigate to parent
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
            if not isinstance(target, dict):
                raise ValueError(f"Cannot set {path}: {part} is not a dictionary")

        # Set value
        target[parts[-1]] = value

        # Invalidate cache for this path and parent paths
        for i in range(len(parts)):
            cache_path = ".".join(parts[: i + 1])
            self._cache.pop(cache_path, None)

    def add_stage_result(self, result: StageResult) -> None:
        """Add stage execution result.

        Args:
            result: Stage result to add.
        """
        self._stage_results.append(result)
        self._timeline.append(
            {
                "type": "stage_complete",
                "stage": result.stage_name,
                "time_ms": self.elapsed_ms,
                "success": result.error is None,
            }
        )

    def add_stage_error(self, stage_name: str, error: str) -> None:
        """Add stage execution error.

        Args:
            stage_name: Name of the stage that failed.
            error: Error message.
        """
        self._errors.append(f"{stage_name}: {error}")
        self._timeline.append({"type": "stage_error", "stage": stage_name, "time_ms": self.elapsed_ms, "error": error})

    def get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        """Get result for a specific stage.

        Args:
            stage_name: Name of the stage.

        Returns:
            Stage result if found, None otherwise.
        """
        for result in reversed(self._stage_results):
            if result.stage_name == stage_name:
                return result
        return None

    def set_termination_flag(self, reason: Optional[str] = None) -> None:
        """Set termination flag to stop execution.

        Args:
            reason: Optional reason for termination.
        """
        self._termination_flag = True
        self._timeline.append({"type": "termination", "time_ms": self.elapsed_ms, "reason": reason})

    def set_next_stage(self, stage_name: str) -> None:
        """Set next stage to execute (skip_to action).

        Args:
            stage_name: Name of the stage to skip to.
        """
        self._next_stage = stage_name
        self._routing_decisions.append(
            {"type": "skip_to", "target": stage_name, "time_ms": self.elapsed_ms}
        )

    def get_next_stage(self) -> Optional[str]:
        """Get and clear next stage override.

        Returns:
            Next stage name if set, None otherwise.
        """
        stage = self._next_stage
        self._next_stage = None
        return stage

    def enable_stage(self, stage_name: str) -> None:
        """Dynamically enable a stage.

        Args:
            stage_name: Name of the stage to enable.
        """
        self._enabled_stages.add(stage_name)
        self._disabled_stages.discard(stage_name)
        self._routing_decisions.append(
            {"type": "enable_stage", "stage": stage_name, "time_ms": self.elapsed_ms}
        )

    def disable_stage(self, stage_name: str) -> None:
        """Dynamically disable a stage.

        Args:
            stage_name: Name of the stage to disable.
        """
        self._disabled_stages.add(stage_name)
        self._enabled_stages.discard(stage_name)
        self._routing_decisions.append(
            {"type": "disable_stage", "stage": stage_name, "time_ms": self.elapsed_ms}
        )

    def is_stage_enabled(self, stage_name: str, default: bool = True) -> bool:
        """Check if a stage is enabled.

        Args:
            stage_name: Name of the stage.
            default: Default value if not explicitly set.

        Returns:
            True if stage is enabled, False otherwise.
        """
        if stage_name in self._disabled_stages:
            return False
        if stage_name in self._enabled_stages:
            return True
        return default

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value.

        Args:
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            Metadata value or default.
        """
        return self._metadata.get(key, default)

    def get_result(self) -> Dict[str, Any]:
        """Get final execution result.

        Returns a generic result dictionary containing execution summary.

        Returns:
            Dictionary with execution results and metadata.
        """
        # Collect stage results
        stage_results_dict = {}
        for result in self._stage_results:
            stage_results_dict[result.stage_name] = result.to_dict()

        # Build result
        result = {
            "success": not self._errors,
            "execution_time_ms": self.elapsed_ms,
            "stages_executed": len(self._stage_results),
            "stage_results": stage_results_dict,
        }

        # Add errors if any
        if self._errors:
            result["errors"] = self._errors

        # Add metadata if any
        if self._metadata:
            result["metadata"] = self._metadata

        # Add timeline if any events
        if self._timeline:
            result["timeline"] = self._timeline

        # Add routing decisions if any
        if self._routing_decisions:
            result["routing_decisions"] = self._routing_decisions

        return result

    def __repr__(self) -> str:
        return (
            f"ExecutionContext(stages_executed={len(self._stage_results)}, "
            f"errors={len(self._errors)}, elapsed_ms={self.elapsed_ms:.2f})"
        )
