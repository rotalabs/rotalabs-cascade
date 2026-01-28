"""Tests for cascade engine in rotalabs-cascade.

Tests cover:
- CascadeEngine initialization
- Stage registration
- Execution with simple config
- Routing rules and stage enabling
- Termination conditions
- Timeout handling
- Statistics tracking
"""

import asyncio
import pytest

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


class TestEngineInitialization:
    """Tests for engine initialization."""

    def test_engine_creation(self, simple_config):
        """Test creating a cascade engine."""
        engine = CascadeEngine(simple_config)

        assert engine.config == simple_config
        assert len(engine._handlers) == 0
        assert len(engine._compiled_rules) > 0

    def test_engine_compiles_global_rules(self):
        """Test that engine compiles global termination conditions."""
        config = CascadeConfig(
            name="test",
            version="1.0.0",
            stages={"STAGE": StageConfig(name="STAGE")},
            global_termination_conditions=[
                Condition(
                    field="abort",
                    operator=ConditionOperator.EQ,
                    value=True,
                )
            ],
        )

        engine = CascadeEngine(config)

        assert "__global__" in engine._compiled_rules
        assert len(engine._compiled_rules["__global__"]) > 0

    def test_engine_compiles_stage_rules(self, simple_config):
        """Test that engine compiles stage-specific rules."""
        engine = CascadeEngine(simple_config)

        assert "FAST" in engine._compiled_rules
        assert len(engine._compiled_rules["FAST"]) > 0


class TestStageRegistration:
    """Tests for stage registration."""

    @pytest.mark.asyncio
    async def test_register_handler(self, simple_config, async_handler):
        """Test registering a stage handler."""
        engine = CascadeEngine(simple_config)
        handler = async_handler()

        engine.register_stage("FAST", handler)

        assert "FAST" in engine._handlers

    @pytest.mark.asyncio
    async def test_register_nonexistent_stage(self, simple_config, async_handler):
        """Test that registering handler for unknown stage fails."""
        engine = CascadeEngine(simple_config)
        handler = async_handler()

        with pytest.raises(ValueError, match="not found in configuration"):
            engine.register_stage("NONEXISTENT", handler)

    @pytest.mark.asyncio
    async def test_handler_monitoring(self, simple_config, async_handler):
        """Test that registered handlers are wrapped with monitoring."""
        engine = CascadeEngine(simple_config)
        handler = async_handler(result="test_result")

        engine.register_stage("FAST", handler)

        # Execute handler
        context = ExecutionContext({"data": "test"})
        await engine._handlers["FAST"](context)

        # Check statistics were updated
        stats = engine.get_statistics()
        assert "FAST" in stats
        assert stats["FAST"]["count"] == 1


class TestExecution:
    """Tests for cascade execution."""

    @pytest.mark.asyncio
    async def test_simple_execution(self, simple_config, async_handler, sample_data):
        """Test executing a simple cascade."""
        engine = CascadeEngine(simple_config)

        # Register handlers
        engine.register_stage("FAST", async_handler(result="fast_result", confidence=0.9))
        engine.register_stage("MEDIUM", async_handler(result="medium_result"))
        engine.register_stage("SLOW", async_handler(result="slow_result"))

        result = await engine.execute(sample_data)

        assert result["success"] is True
        assert result["stages_executed"] >= 1

    @pytest.mark.asyncio
    async def test_execution_order(self, sample_data):
        """Test that stages execute in correct order."""
        config = CascadeConfig(
            name="ordered",
            version="1.0.0",
            stages={
                "FIRST": StageConfig(name="FIRST"),
                "SECOND": StageConfig(name="SECOND"),
                "THIRD": StageConfig(name="THIRD"),
            },
            execution_order=["FIRST", "SECOND", "THIRD"],
        )

        engine = CascadeEngine(config)

        execution_order = []

        async def create_handler(name):
            async def handler(context):
                execution_order.append(name)
                return {"result": f"{name}_done"}
            return handler

        engine.register_stage("FIRST", await create_handler("FIRST"))
        engine.register_stage("SECOND", await create_handler("SECOND"))
        engine.register_stage("THIRD", await create_handler("THIRD"))

        await engine.execute(sample_data)

        assert execution_order == ["FIRST", "SECOND", "THIRD"]

    @pytest.mark.asyncio
    async def test_execution_with_disabled_stage(self, sample_data):
        """Test that disabled stages are skipped."""
        config = CascadeConfig(
            name="test",
            version="1.0.0",
            stages={
                "ENABLED": StageConfig(name="ENABLED", enabled=True),
                "DISABLED": StageConfig(name="DISABLED", enabled=False),
            },
            execution_order=["ENABLED", "DISABLED"],
        )

        engine = CascadeEngine(config)

        executed = []

        async def tracking_handler(name):
            async def handler(context):
                executed.append(name)
                return {"result": "done"}
            return handler

        engine.register_stage("ENABLED", await tracking_handler("ENABLED"))
        engine.register_stage("DISABLED", await tracking_handler("DISABLED"))

        result = await engine.execute(sample_data)

        assert "ENABLED" in executed
        assert "DISABLED" not in executed

    @pytest.mark.asyncio
    async def test_dependency_handling(self, sample_data):
        """Test that stage dependencies are respected."""
        config = CascadeConfig(
            name="deps",
            version="1.0.0",
            stages={
                "INDEPENDENT": StageConfig(name="INDEPENDENT"),
                "DEPENDENT": StageConfig(
                    name="DEPENDENT",
                    depends_on=["INDEPENDENT"],
                ),
            },
            execution_order=["DEPENDENT", "INDEPENDENT"],  # Wrong order
        )

        engine = CascadeEngine(config)

        execution_order = []

        async def create_handler(name):
            async def handler(context):
                execution_order.append(name)
                return {"result": "done"}
            return handler

        engine.register_stage("INDEPENDENT", await create_handler("INDEPENDENT"))
        engine.register_stage("DEPENDENT", await create_handler("DEPENDENT"))

        await engine.execute(sample_data)

        # INDEPENDENT should execute before DEPENDENT despite wrong order
        assert execution_order.index("INDEPENDENT") < execution_order.index("DEPENDENT")


class TestRoutingRules:
    """Tests for routing rules and stage enabling."""

    @pytest.mark.asyncio
    async def test_enable_stages_action(self, sample_data):
        """Test that enable_stages action enables stages."""
        config = CascadeConfig(
            name="routing",
            version="1.0.0",
            stages={
                "TRIGGER": StageConfig(
                    name="TRIGGER",
                    routing_rules=[
                        RoutingRule(
                            name="enable_next",
                            type="routing",
                            condition=Condition(
                                field="enable_next",
                                operator=ConditionOperator.EQ,
                                value=True,
                            ),
                            action=RoutingAction(
                                type="enable_stages",
                                stages=["NEXT"],
                            ),
                        )
                    ],
                ),
                "NEXT": StageConfig(name="NEXT", enabled=False),
            },
            execution_order=["TRIGGER", "NEXT"],
        )

        engine = CascadeEngine(config)

        executed = []

        async def tracking_handler(name):
            async def handler(context):
                executed.append(name)
                return {"result": "done"}
            return handler

        engine.register_stage("TRIGGER", await tracking_handler("TRIGGER"))
        engine.register_stage("NEXT", await tracking_handler("NEXT"))

        # Execute with enable_next=True
        data = {**sample_data, "enable_next": True}
        await engine.execute(data)

        assert "TRIGGER" in executed
        assert "NEXT" in executed

    @pytest.mark.asyncio
    async def test_disable_stages_action(self, sample_data):
        """Test that disable_stages action disables stages."""
        config = CascadeConfig(
            name="routing",
            version="1.0.0",
            stages={
                "FIRST": StageConfig(
                    name="FIRST",
                    routing_rules=[
                        RoutingRule(
                            name="disable_second",
                            type="routing",
                            condition=Condition(
                                field="skip_second",
                                operator=ConditionOperator.EQ,
                                value=True,
                            ),
                            action=RoutingAction(
                                type="disable_stages",
                                stages=["SECOND"],
                            ),
                        )
                    ],
                ),
                "SECOND": StageConfig(name="SECOND", enabled=True),
            },
            execution_order=["FIRST", "SECOND"],
        )

        engine = CascadeEngine(config)

        executed = []

        async def tracking_handler(name):
            async def handler(context):
                executed.append(name)
                return {"result": "done"}
            return handler

        engine.register_stage("FIRST", await tracking_handler("FIRST"))
        engine.register_stage("SECOND", await tracking_handler("SECOND"))

        # Execute with skip_second=True
        data = {**sample_data, "skip_second": True}
        await engine.execute(data)

        assert "FIRST" in executed
        assert "SECOND" not in executed

    @pytest.mark.asyncio
    async def test_skip_to_action(self, sample_data):
        """Test that skip_to action jumps to target stage."""
        config = CascadeConfig(
            name="routing",
            version="1.0.0",
            stages={
                "FIRST": StageConfig(
                    name="FIRST",
                    routing_rules=[
                        RoutingRule(
                            name="skip_to_third",
                            type="routing",
                            condition=Condition(
                                field="skip_middle",
                                operator=ConditionOperator.EQ,
                                value=True,
                            ),
                            action=RoutingAction(
                                type="skip_to",
                                target="THIRD",
                            ),
                        )
                    ],
                ),
                "SECOND": StageConfig(name="SECOND"),
                "THIRD": StageConfig(name="THIRD"),
            },
            execution_order=["FIRST", "SECOND", "THIRD"],
        )

        engine = CascadeEngine(config)

        executed = []

        async def tracking_handler(name):
            async def handler(context):
                executed.append(name)
                return {"result": "done"}
            return handler

        engine.register_stage("FIRST", await tracking_handler("FIRST"))
        engine.register_stage("SECOND", await tracking_handler("SECOND"))
        engine.register_stage("THIRD", await tracking_handler("THIRD"))

        # Execute with skip_middle=True
        data = {**sample_data, "skip_middle": True}
        await engine.execute(data)

        assert "FIRST" in executed
        assert "SECOND" not in executed
        assert "THIRD" in executed


class TestTermination:
    """Tests for termination conditions."""

    @pytest.mark.asyncio
    async def test_global_termination(self, sample_data):
        """Test global termination condition."""
        config = CascadeConfig(
            name="termination",
            version="1.0.0",
            stages={
                "FIRST": StageConfig(name="FIRST"),
                "SECOND": StageConfig(name="SECOND"),
            },
            execution_order=["FIRST", "SECOND"],
            global_termination_conditions=[
                Condition(
                    field="abort",
                    operator=ConditionOperator.EQ,
                    value=True,
                )
            ],
        )

        engine = CascadeEngine(config)

        executed = []

        async def tracking_handler(name):
            async def handler(context):
                executed.append(name)
                return {"result": "done"}
            return handler

        engine.register_stage("FIRST", await tracking_handler("FIRST"))
        engine.register_stage("SECOND", await tracking_handler("SECOND"))

        # Execute with abort=True
        data = {**sample_data, "abort": True}
        await engine.execute(data)

        # Both stages should be skipped
        assert len(executed) == 0

    @pytest.mark.asyncio
    async def test_terminate_action(self, sample_data):
        """Test terminate action stops execution."""
        config = CascadeConfig(
            name="termination",
            version="1.0.0",
            stages={
                "FIRST": StageConfig(
                    name="FIRST",
                    routing_rules=[
                        RoutingRule(
                            name="terminate",
                            type="routing",
                            condition=Condition(
                                field="stop_now",
                                operator=ConditionOperator.EQ,
                                value=True,
                            ),
                            action=RoutingAction(type="terminate"),
                        )
                    ],
                ),
                "SECOND": StageConfig(name="SECOND"),
            },
            execution_order=["FIRST", "SECOND"],
        )

        engine = CascadeEngine(config)

        executed = []

        async def tracking_handler(name):
            async def handler(context):
                executed.append(name)
                return {"result": "done"}
            return handler

        engine.register_stage("FIRST", await tracking_handler("FIRST"))
        engine.register_stage("SECOND", await tracking_handler("SECOND"))

        # Execute with stop_now=True
        data = {**sample_data, "stop_now": True}
        await engine.execute(data)

        assert "FIRST" in executed
        assert "SECOND" not in executed


class TestTimeout:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_stage_timeout(self, sample_data):
        """Test that stage timeout is enforced."""
        config = CascadeConfig(
            name="timeout",
            version="1.0.0",
            stages={
                "SLOW": StageConfig(name="SLOW", timeout_ms=100),
            },
        )

        engine = CascadeEngine(config)

        async def slow_handler(context):
            await asyncio.sleep(1)  # Sleep 1 second (exceeds 100ms timeout)
            return {"result": "done"}

        engine.register_stage("SLOW", slow_handler)

        result = await engine.execute(sample_data)

        # Should have error due to timeout
        assert result["success"] is False
        assert len(result.get("errors", [])) > 0

    @pytest.mark.asyncio
    async def test_global_timeout(self, sample_data):
        """Test that global timeout is enforced."""
        config = CascadeConfig(
            name="timeout",
            version="1.0.0",
            stages={
                "SLOW": StageConfig(name="SLOW"),
            },
            global_timeout_ms=100,
        )

        engine = CascadeEngine(config)

        async def slow_handler(context):
            await asyncio.sleep(1)  # Exceeds global timeout
            return {"result": "done"}

        engine.register_stage("SLOW", slow_handler)

        with pytest.raises(TimeoutError):
            await engine.execute(sample_data)


class TestRetry:
    """Tests for retry logic."""

    @pytest.mark.asyncio
    async def test_stage_retry(self, sample_data):
        """Test that stages retry on failure."""
        config = CascadeConfig(
            name="retry",
            version="1.0.0",
            stages={
                "FLAKY": StageConfig(
                    name="FLAKY",
                    max_retries=2,
                    retry_delay_ms=10,
                ),
            },
        )

        engine = CascadeEngine(config)

        attempt_count = 0

        async def flaky_handler(context):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise RuntimeError("Temporary failure")
            return {"result": "success"}

        engine.register_stage("FLAKY", flaky_handler)

        result = await engine.execute(sample_data)

        assert result["success"] is True
        assert attempt_count == 2  # First attempt + 1 retry


class TestCaching:
    """Tests for result caching."""

    @pytest.mark.asyncio
    async def test_stage_result_caching(self, sample_data):
        """Test that stage results are cached."""
        config = CascadeConfig(
            name="caching",
            version="1.0.0",
            stages={
                "CACHED": StageConfig(
                    name="CACHED",
                    cache_enabled=True,
                    cache_ttl_seconds=60,
                ),
            },
            enable_caching=True,
        )

        engine = CascadeEngine(config)

        call_count = 0

        async def counted_handler(context):
            nonlocal call_count
            call_count += 1
            return {"result": f"call_{call_count}"}

        engine.register_stage("CACHED", counted_handler)

        # First execution
        result1 = await engine.execute(sample_data)
        assert call_count == 1

        # Second execution should use cache
        result2 = await engine.execute(sample_data)
        assert call_count == 1  # Handler not called again


class TestStatistics:
    """Tests for execution statistics."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, simple_config, async_handler, sample_data):
        """Test retrieving execution statistics."""
        engine = CascadeEngine(simple_config)

        engine.register_stage("FAST", async_handler())
        engine.register_stage("MEDIUM", async_handler())
        engine.register_stage("SLOW", async_handler())

        await engine.execute(sample_data)

        stats = engine.get_statistics()

        assert "FAST" in stats
        assert stats["FAST"]["count"] >= 1
        assert stats["FAST"]["total_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_statistics_track_errors(self, sample_data):
        """Test that statistics track errors."""
        config = CascadeConfig(
            name="stats",
            version="1.0.0",
            stages={
                "FAILING": StageConfig(name="FAILING", max_retries=0),
            },
        )

        engine = CascadeEngine(config)

        async def failing_handler(context):
            raise RuntimeError("Intentional failure")

        engine.register_stage("FAILING", failing_handler)

        await engine.execute(sample_data)

        stats = engine.get_statistics()

        assert stats["FAILING"]["errors"] >= 1

    def test_clear_cache(self, simple_config):
        """Test clearing engine caches."""
        engine = CascadeEngine(simple_config)

        # Add some data to caches
        engine._result_cache["key1"] = ("value1", 123.0)
        engine._plan_cache["key2"] = ["STAGE_A"]

        engine.clear_cache()

        assert len(engine._result_cache) == 0
        assert len(engine._plan_cache) == 0

    def test_update_config(self, simple_config):
        """Test hot-reloading configuration."""
        engine = CascadeEngine(simple_config)

        # Create new config
        new_config = CascadeConfig(
            name="updated",
            version="2.0.0",
            stages={"NEW_STAGE": StageConfig(name="NEW_STAGE")},
        )

        engine.update_config(new_config)

        assert engine.config.name == "updated"
        assert engine.config.version == "2.0.0"
        assert "NEW_STAGE" in engine.config.stages
