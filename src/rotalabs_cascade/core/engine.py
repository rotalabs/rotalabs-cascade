"""Cascade orchestration engine.

This module provides the main execution engine for configuration-driven
multi-stage decision routing and orchestration.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from rotalabs_cascade.core.config import (
    CascadeConfig,
    Condition,
    ConditionOperator,
    RoutingAction,
    RoutingRule,
    StageConfig,
)
from rotalabs_cascade.core.context import ExecutionContext, StageResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CascadeEngine:
    """Main orchestration engine for cascade execution.

    The engine manages stage registration, execution planning, and dynamic
    routing based on configuration and runtime conditions.

    Uses __slots__ for memory efficiency.

    Attributes:
        config: Cascade configuration.
        _handlers: Registered stage handlers.
        _result_cache: Cache for stage results.
        _plan_cache: Cache for execution plans.
        _statistics: Execution statistics.
    """

    __slots__ = (
        "config",
        "_handlers",
        "_result_cache",
        "_plan_cache",
        "_statistics",
        "_compiled_rules",
    )

    def __init__(self, config: CascadeConfig):
        """Initialize cascade engine.

        Args:
            config: Cascade configuration.
        """
        self.config = config
        self._handlers: Dict[str, Callable] = {}
        self._result_cache: Dict[str, Tuple[Any, float]] = {}  # key -> (result, timestamp)
        self._plan_cache: Dict[str, List[str]] = {}  # cache_key -> execution_plan
        self._statistics = defaultdict(lambda: {"count": 0, "total_time_ms": 0, "errors": 0})
        self._compiled_rules: Dict[str, List[RoutingRule]] = {}

        # Compile routing rules for faster lookup
        self._compile_routing_rules()

        logger.info(f"Initialized CascadeEngine: {config.name} v{config.version}")
        logger.info(f"Configured stages: {list(config.stages.keys())}")

    def _compile_routing_rules(self) -> None:
        """Compile and organize routing rules for efficient lookup."""
        # Global termination conditions
        if self.config.global_termination_conditions:
            self._compiled_rules["__global__"] = [
                RoutingRule(
                    name=f"global_termination_{i}",
                    type="precondition",
                    condition=cond,
                    action=RoutingAction(type="terminate"),
                    priority=1000,
                )
                for i, cond in enumerate(self.config.global_termination_conditions)
            ]
        else:
            self._compiled_rules["__global__"] = []

        # Stage-specific rules
        for stage_name, stage in self.config.stages.items():
            if stage.routing_rules:
                # Sort by priority (descending)
                self._compiled_rules[stage_name] = sorted(stage.routing_rules, key=lambda r: -r.priority)
            else:
                self._compiled_rules[stage_name] = []

    def register_stage(self, name: str, handler: Callable) -> None:
        """Register a stage handler.

        Wraps handler with monitoring and error handling.

        Args:
            name: Stage name matching configuration.
            handler: Async callable that processes stage execution.

        Raises:
            ValueError: If stage name not in configuration.
        """
        if name not in self.config.stages:
            raise ValueError(f"Stage {name} not found in configuration")

        # Wrap handler with monitoring
        async def monitored_handler(context: ExecutionContext) -> Any:
            stage_start = time.time()
            try:
                result = await handler(context)
                self._statistics[name]["count"] += 1
                self._statistics[name]["total_time_ms"] += (time.time() - stage_start) * 1000
                return result
            except Exception as e:
                self._statistics[name]["errors"] += 1
                logger.error(f"Stage {name} failed: {e}", exc_info=True)
                raise

        self._handlers[name] = monitored_handler
        logger.info(f"Registered handler for stage: {name}")

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cascade with input data.

        Main entry point for cascade execution. Creates execution context,
        generates or retrieves cached execution plan, and executes stages
        with dynamic routing.

        Args:
            data: Input data dictionary.

        Returns:
            Execution result dictionary.

        Raises:
            TimeoutError: If global timeout exceeded.
            RuntimeError: If execution fails critically.
        """
        logger.info(f"Starting cascade execution: {self.config.name}")

        # Create execution context
        context = ExecutionContext(data)

        # Generate or retrieve execution plan
        cache_key = self._get_cache_key(data) if self.config.enable_caching else None
        plan = self._plan_cache.get(cache_key) if cache_key else None

        if plan is None:
            plan = self._generate_execution_plan()
            if cache_key:
                self._plan_cache[cache_key] = plan

        logger.info(f"Execution plan: {plan}")

        # Execute plan with timeout
        try:
            async with asyncio.timeout(self.config.global_timeout_ms / 1000):
                await self._execute_plan(plan, context)
        except asyncio.TimeoutError:
            logger.error(f"Global timeout exceeded: {self.config.global_timeout_ms}ms")
            context.add_stage_error("__global__", "Global timeout exceeded")
            raise TimeoutError(f"Cascade execution exceeded {self.config.global_timeout_ms}ms")

        # Get final result
        result = context.get_result()
        logger.info(
            f"Cascade execution complete: {result['stages_executed']} stages in {result['execution_time_ms']:.2f}ms"
        )

        return result

    def _generate_execution_plan(self) -> List[str]:
        """Generate execution plan from configuration.

        Uses execution_order if provided, otherwise performs topological sort
        based on stage dependencies.

        Returns:
            Ordered list of stage names.
        """
        if self.config.execution_order:
            return self.config.execution_order.copy()

        # Topological sort based on dependencies
        stages = self.config.stages
        in_degree = {name: 0 for name in stages}
        graph = defaultdict(list)

        # Build dependency graph
        for name, stage in stages.items():
            for dep in stage.depends_on:
                graph[dep].append(name)
                in_degree[name] += 1

        # Kahn's algorithm
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        plan = []

        while queue:
            node = queue.popleft()
            plan.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(plan) != len(stages):
            raise RuntimeError("Circular dependency detected in stage configuration")

        return plan

    async def _execute_plan(self, plan: List[str], context: ExecutionContext) -> None:
        """Execute stages according to plan with dynamic routing.

        Uses a queue-based approach to handle dynamic stage enabling/disabling
        and skip_to actions.

        Args:
            plan: Ordered list of stage names to execute.
            context: Execution context.
        """
        # Initialize queue with plan
        queue = deque(plan)
        executed = set()

        while queue and not context.should_terminate:
            # Check global termination conditions
            if self._should_terminate(context):
                logger.info("Global termination condition met")
                context.set_termination_flag("Global termination condition")
                break

            # Check for skip_to override
            next_stage = context.get_next_stage()
            if next_stage:
                logger.info(f"Skipping to stage: {next_stage}")
                # Remove stages until we find the target
                while queue and queue[0] != next_stage:
                    queue.popleft()
                if not queue:
                    logger.warning(f"Skip target {next_stage} not found in remaining queue")
                    break

            # Get next stage
            if not queue:
                break
            stage_name = queue.popleft()

            # Check if stage is enabled
            stage_config = self.config.stages[stage_name]
            if not context.is_stage_enabled(stage_name, stage_config.enabled):
                logger.info(f"Skipping disabled stage: {stage_name}")
                continue

            # Check if already executed (avoid duplicates)
            if stage_name in executed:
                logger.info(f"Skipping already executed stage: {stage_name}")
                continue

            # Check dependencies
            if not self._check_dependencies(stage_name, executed):
                logger.info(f"Dependencies not met for stage: {stage_name}, re-queuing")
                queue.append(stage_name)  # Re-queue for later
                continue

            # Check preconditions
            if not await self._check_preconditions(stage_name, context):
                logger.info(f"Preconditions not met for stage: {stage_name}")
                continue

            # Execute stage
            try:
                await self._execute_stage(stage_name, context)
                executed.add(stage_name)

                # Evaluate routing rules
                newly_enabled = await self._evaluate_routing(stage_name, context)

                # Add newly enabled stages to queue
                for new_stage in newly_enabled:
                    if new_stage not in executed and new_stage not in queue:
                        queue.append(new_stage)
                        logger.info(f"Added newly enabled stage to queue: {new_stage}")

            except Exception as e:
                logger.error(f"Stage {stage_name} execution failed: {e}", exc_info=True)
                context.add_stage_error(stage_name, str(e))
                # Continue with next stage unless it's a critical failure

    def _check_dependencies(self, stage_name: str, executed: Set[str]) -> bool:
        """Check if stage dependencies are satisfied.

        Args:
            stage_name: Name of the stage.
            executed: Set of executed stage names.

        Returns:
            True if dependencies are satisfied, False otherwise.
        """
        stage = self.config.stages[stage_name]
        return all(dep in executed for dep in stage.depends_on)

    async def _check_preconditions(self, stage_name: str, context: ExecutionContext) -> bool:
        """Check stage preconditions.

        Args:
            stage_name: Name of the stage.
            context: Execution context.

        Returns:
            True if preconditions are met, False otherwise.
        """
        rules = self._compiled_rules.get(stage_name, [])
        preconditions = [r for r in rules if r.type == "precondition"]

        for rule in preconditions:
            if self._evaluate_condition(rule.condition, context):
                logger.info(f"Precondition failed for {stage_name}: {rule.name}")
                # Execute action (typically terminate or skip)
                self._execute_action(rule.action, context)
                return False

        return True

    async def _evaluate_routing(self, stage_name: str, context: ExecutionContext) -> List[str]:
        """Evaluate routing rules after stage execution.

        Args:
            stage_name: Name of the executed stage.
            context: Execution context.

        Returns:
            List of newly enabled stage names.
        """
        rules = self._compiled_rules.get(stage_name, [])
        routing_rules = [r for r in rules if r.type == "routing"]
        newly_enabled = []

        for rule in routing_rules:
            if self._evaluate_condition(rule.condition, context):
                logger.info(f"Routing rule matched for {stage_name}: {rule.name}")
                enabled = self._execute_action(rule.action, context)
                if enabled:
                    newly_enabled.extend(enabled)

        # Also evaluate postconditions
        postconditions = [r for r in rules if r.type == "postcondition"]
        for rule in postconditions:
            if self._evaluate_condition(rule.condition, context):
                logger.info(f"Postcondition matched for {stage_name}: {rule.name}")
                self._execute_action(rule.action, context)

        return newly_enabled

    def _evaluate_condition(self, condition: Condition, context: ExecutionContext) -> bool:
        """Evaluate a condition against execution context.

        Args:
            condition: Condition to evaluate.
            context: Execution context.

        Returns:
            True if condition is met, False otherwise.
        """
        op = condition.operator

        # Logical operators
        if op == ConditionOperator.AND:
            return all(self._evaluate_condition(c, context) for c in condition.conditions or [])
        elif op == ConditionOperator.OR:
            return any(self._evaluate_condition(c, context) for c in condition.conditions or [])
        elif op == ConditionOperator.NOT:
            return not self._evaluate_condition(condition.conditions[0], context) if condition.conditions else False

        # Get field value
        field_value = context.get(condition.field)

        # Existence operators
        if op == ConditionOperator.EXISTS:
            return field_value is not None
        elif op == ConditionOperator.IS_NULL:
            return field_value is None

        # If field doesn't exist, condition fails (except for EXISTS/IS_NULL)
        if field_value is None:
            return False

        # Comparison operators
        if op == ConditionOperator.EQ:
            return field_value == condition.value
        elif op == ConditionOperator.NE:
            return field_value != condition.value
        elif op == ConditionOperator.GT:
            return field_value > condition.value
        elif op == ConditionOperator.GE:
            return field_value >= condition.value
        elif op == ConditionOperator.LT:
            return field_value < condition.value
        elif op == ConditionOperator.LE:
            return field_value <= condition.value

        # Collection operators
        elif op == ConditionOperator.IN:
            return field_value in condition.value
        elif op == ConditionOperator.NOT_IN:
            return field_value not in condition.value
        elif op == ConditionOperator.CONTAINS:
            return condition.value in field_value

        # Pattern matching
        elif op == ConditionOperator.MATCHES:
            return bool(re.match(condition.value, str(field_value)))

        # Aggregation operators (for lists)
        elif op == ConditionOperator.ALL:
            if not isinstance(field_value, list):
                return False
            return all(item == condition.value for item in field_value)
        elif op == ConditionOperator.ANY:
            if not isinstance(field_value, list):
                return False
            return any(item == condition.value for item in field_value)
        elif op == ConditionOperator.NONE:
            if not isinstance(field_value, list):
                return False
            return not any(item == condition.value for item in field_value)

        # Statistical operators
        elif op == ConditionOperator.SUM:
            if not isinstance(field_value, list):
                return False
            return sum(field_value) == condition.value
        elif op == ConditionOperator.AVG:
            if not isinstance(field_value, list) or not field_value:
                return False
            return sum(field_value) / len(field_value) == condition.value
        elif op == ConditionOperator.MIN:
            if not isinstance(field_value, list):
                return False
            return min(field_value) == condition.value
        elif op == ConditionOperator.MAX:
            if not isinstance(field_value, list):
                return False
            return max(field_value) == condition.value
        elif op == ConditionOperator.COUNT:
            if not isinstance(field_value, list):
                return False
            return len(field_value) == condition.value

        logger.warning(f"Unknown operator: {op}")
        return False

    def _execute_action(self, action: RoutingAction, context: ExecutionContext) -> Optional[List[str]]:
        """Execute a routing action.

        Args:
            action: Action to execute.
            context: Execution context.

        Returns:
            List of newly enabled stage names if applicable, None otherwise.
        """
        if action.type == "terminate":
            context.set_termination_flag(f"Action: {action.type}")

        elif action.type == "skip_to":
            context.set_next_stage(action.target)

        elif action.type == "enable_stages":
            for stage in action.stages or []:
                context.enable_stage(stage)
            return action.stages

        elif action.type == "disable_stages":
            for stage in action.stages or []:
                context.disable_stage(stage)

        elif action.type == "set_field":
            context.set(action.field, action.value)

        return None

    def _should_terminate(self, context: ExecutionContext) -> bool:
        """Check global termination conditions.

        Args:
            context: Execution context.

        Returns:
            True if should terminate, False otherwise.
        """
        global_rules = self._compiled_rules.get("__global__", [])
        for rule in global_rules:
            if self._evaluate_condition(rule.condition, context):
                logger.info(f"Global termination condition met: {rule.name}")
                return True
        return False

    async def _execute_stage(self, stage_name: str, context: ExecutionContext) -> None:
        """Execute a single stage with timeout and retry logic.

        Args:
            stage_name: Name of the stage to execute.
            context: Execution context.

        Raises:
            RuntimeError: If stage handler not registered.
        """
        stage_config = self.config.stages[stage_name]

        # Check if handler registered
        if stage_name not in self._handlers:
            raise RuntimeError(f"No handler registered for stage: {stage_name}")

        logger.info(f"Executing stage: {stage_name}")

        # Check result cache
        if stage_config.cache_enabled and self.config.enable_caching:
            cache_key = self._get_stage_cache_key(stage_name, context)
            cached = self._result_cache.get(cache_key)
            if cached:
                result, timestamp = cached
                age_seconds = time.time() - timestamp
                if age_seconds < stage_config.cache_ttl_seconds:
                    logger.info(f"Using cached result for {stage_name} (age: {age_seconds:.1f}s)")
                    context.add_stage_result(
                        StageResult(
                            stage_name=stage_name,
                            result=result,
                            data={"cached": True, "age_seconds": age_seconds},
                            time_ms=0,
                        )
                    )
                    return

        # Execute with retry logic
        handler = self._handlers[stage_name]
        last_error = None

        for attempt in range(stage_config.max_retries + 1):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt} for {stage_name}")
                await asyncio.sleep(stage_config.retry_delay_ms / 1000)

            start_time = time.time()
            try:
                # Execute with timeout
                async with asyncio.timeout(stage_config.timeout_ms / 1000):
                    result = await handler(context)

                elapsed_ms = (time.time() - start_time) * 1000

                # Create stage result
                stage_result = StageResult(
                    stage_name=stage_name,
                    result=result.get("result") if isinstance(result, dict) else result,
                    confidence=result.get("confidence") if isinstance(result, dict) else None,
                    data=result.get("data", {}) if isinstance(result, dict) else {},
                    time_ms=elapsed_ms,
                )

                context.add_stage_result(stage_result)

                # Cache result if enabled
                if stage_config.cache_enabled and self.config.enable_caching:
                    cache_key = self._get_stage_cache_key(stage_name, context)
                    self._result_cache[cache_key] = (result, time.time())

                logger.info(f"Stage {stage_name} completed in {elapsed_ms:.2f}ms")
                return

            except asyncio.TimeoutError:
                last_error = f"Timeout after {stage_config.timeout_ms}ms"
                logger.warning(f"Stage {stage_name} timed out (attempt {attempt + 1})")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Stage {stage_name} failed (attempt {attempt + 1}): {e}", exc_info=True)

        # All retries exhausted
        elapsed_ms = (time.time() - start_time) * 1000
        context.add_stage_result(
            StageResult(stage_name=stage_name, error=last_error, time_ms=elapsed_ms)
        )
        context.add_stage_error(stage_name, last_error)

    def _get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from data.

        Args:
            data: Input data dictionary.

        Returns:
            Cache key string.
        """
        if not self.config.cache_key_fields:
            # Use hash of entire data
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(data_str.encode()).hexdigest()

        # Use specified fields
        key_data = {field: data.get(field) for field in self.config.cache_key_fields}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_stage_cache_key(self, stage_name: str, context: ExecutionContext) -> str:
        """Generate cache key for stage result.

        Args:
            stage_name: Name of the stage.
            context: Execution context.

        Returns:
            Cache key string.
        """
        base_key = self._get_cache_key(context.data)
        return f"{stage_name}:{base_key}"

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary of statistics per stage.
        """
        return dict(self._statistics)

    def clear_cache(self) -> None:
        """Clear all caches (result and plan caches)."""
        self._result_cache.clear()
        self._plan_cache.clear()
        logger.info("Caches cleared")

    def update_config(self, config: CascadeConfig) -> None:
        """Hot-reload configuration.

        Updates engine configuration and recompiles routing rules without
        losing registered handlers or statistics.

        Args:
            config: New cascade configuration.
        """
        logger.info(f"Updating configuration: {config.name} v{config.version}")
        self.config = config
        self._compile_routing_rules()
        self.clear_cache()  # Clear caches as execution plan may change
        logger.info("Configuration updated successfully")
