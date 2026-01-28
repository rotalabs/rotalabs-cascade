"""Tests for plugins in rotalabs-cascade.

Tests cover:
- PluginRegistry register/get operations
- CachePlugin caching behavior and TTL
- RetryPlugin retry logic with exponential backoff
- MetricsPlugin metrics tracking
- CircuitBreakerPlugin state transitions
- PluginFactory handler wrapping
"""

import asyncio
import pytest
import time

from rotalabs_cascade.plugins.builtin import (
    CachePlugin,
    CircuitBreakerPlugin,
    MetricsPlugin,
    PluginFactory,
    PluginRegistry,
    RetryPlugin,
    StagePlugin,
)


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_registry_initialization(self):
        """Test creating a plugin registry."""
        registry = PluginRegistry()

        assert len(registry._plugins) == 0
        assert len(registry._handlers) == 0

    def test_register_plugin(self):
        """Test registering a plugin."""
        registry = PluginRegistry()

        async def dummy_handler(context):
            return {"result": "done"}

        plugin = CachePlugin(dummy_handler, ttl_seconds=300)
        registry.register_plugin(plugin)

        assert "cache" in registry._plugins
        assert registry.get_plugin("cache") is plugin

    def test_register_handler(self):
        """Test registering a handler."""
        registry = PluginRegistry()

        async def test_handler(context):
            return {"result": "test"}

        registry.register_handler("test", test_handler)

        assert "test" in registry._handlers
        assert registry.get_handler("test") is test_handler

    def test_get_nonexistent_plugin(self):
        """Test getting non-existent plugin returns None."""
        registry = PluginRegistry()

        assert registry.get_plugin("nonexistent") is None

    def test_get_nonexistent_handler(self):
        """Test getting non-existent handler returns None."""
        registry = PluginRegistry()

        assert registry.get_handler("nonexistent") is None

    def test_list_plugins(self):
        """Test listing registered plugins."""
        registry = PluginRegistry()

        async def dummy_handler(context):
            return {}

        plugin1 = CachePlugin(dummy_handler)
        plugin2 = MetricsPlugin(dummy_handler)

        registry.register_plugin(plugin1)
        registry.register_plugin(plugin2)

        plugins = registry.list_plugins()

        assert len(plugins) == 2
        plugin_names = [p["name"] for p in plugins]
        assert "cache" in plugin_names
        assert "metrics" in plugin_names

    def test_overwrite_plugin(self):
        """Test that registering a plugin twice overwrites."""
        registry = PluginRegistry()

        async def handler1(context):
            return {"v": 1}

        async def handler2(context):
            return {"v": 2}

        plugin1 = CachePlugin(handler1, ttl_seconds=100)
        plugin2 = CachePlugin(handler2, ttl_seconds=200)

        registry.register_plugin(plugin1)
        registry.register_plugin(plugin2)

        retrieved = registry.get_plugin("cache")
        assert retrieved.ttl_seconds == 200


class TestCachePlugin:
    """Tests for CachePlugin."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit returns cached result."""
        call_count = 0

        async def handler(context):
            nonlocal call_count
            call_count += 1
            return {"result": f"call_{call_count}"}

        plugin = CachePlugin(handler, ttl_seconds=60)

        context = {"input": "test"}

        # First call - cache miss
        result1 = await plugin.execute(context)
        assert result1["result"] == "call_1"
        assert call_count == 1

        # Second call - cache hit
        result2 = await plugin.execute(context)
        assert result2["result"] == "call_1"  # Same result
        assert call_count == 1  # Handler not called again

    @pytest.mark.asyncio
    async def test_cache_miss_different_context(self):
        """Test different contexts result in cache miss."""
        call_count = 0

        async def handler(context):
            nonlocal call_count
            call_count += 1
            return {"result": f"call_{call_count}"}

        plugin = CachePlugin(handler, ttl_seconds=60)

        # Different contexts
        result1 = await plugin.execute({"input": "a"})
        result2 = await plugin.execute({"input": "b"})

        assert result1["result"] == "call_1"
        assert result2["result"] == "call_2"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache entries expire after TTL."""
        call_count = 0

        async def handler(context):
            nonlocal call_count
            call_count += 1
            return {"result": f"call_{call_count}"}

        plugin = CachePlugin(handler, ttl_seconds=0.1)  # 100ms TTL

        context = {"input": "test"}

        # First call
        result1 = await plugin.execute(context)
        assert call_count == 1

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Second call - cache expired
        result2 = await plugin.execute(context)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test cache key is generated correctly."""
        async def handler(context):
            return {"result": "done"}

        plugin = CachePlugin(handler)

        context1 = {"a": 1, "b": 2}
        context2 = {"b": 2, "a": 1}  # Same content, different order

        key1 = plugin._get_cache_key(context1)
        key2 = plugin._get_cache_key(context2)

        # Keys should be identical (order-independent)
        assert key1 == key2

    def test_cache_plugin_name(self):
        """Test cache plugin name."""
        async def handler(context):
            return {}

        plugin = CachePlugin(handler)

        assert plugin.name == "cache"


class TestRetryPlugin:
    """Tests for RetryPlugin."""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test no retry needed on success."""
        call_count = 0

        async def handler(context):
            nonlocal call_count
            call_count += 1
            return {"result": "success"}

        plugin = RetryPlugin(handler, max_retries=3, delay_ms=10)

        result = await plugin.execute({"input": "test"})

        assert result["result"] == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test retry succeeds after initial failures."""
        attempt_count = 0

        async def handler(context):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RuntimeError("Temporary failure")
            return {"result": "success"}

        plugin = RetryPlugin(handler, max_retries=3, delay_ms=10)

        result = await plugin.execute({"input": "test"})

        assert result["result"] == "success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test all retries exhausted raises last exception."""
        attempt_count = 0

        async def handler(context):
            nonlocal attempt_count
            attempt_count += 1
            raise RuntimeError(f"Failure {attempt_count}")

        plugin = RetryPlugin(handler, max_retries=2, delay_ms=10)

        with pytest.raises(RuntimeError, match="Failure 3"):
            await plugin.execute({"input": "test"})

        assert attempt_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff(self):
        """Test exponential backoff between retries."""
        attempt_times = []

        async def handler(context):
            attempt_times.append(time.time())
            if len(attempt_times) < 3:
                raise RuntimeError("Fail")
            return {"result": "success"}

        plugin = RetryPlugin(handler, max_retries=3, delay_ms=100)

        await plugin.execute({"input": "test"})

        # Check delays between attempts
        # Delay 1: 100ms, Delay 2: 200ms
        if len(attempt_times) >= 3:
            delay1 = (attempt_times[1] - attempt_times[0]) * 1000
            delay2 = (attempt_times[2] - attempt_times[1]) * 1000

            assert delay1 >= 100
            assert delay2 >= 200
            assert delay2 > delay1  # Exponential increase

    def test_retry_plugin_name(self):
        """Test retry plugin name."""
        async def handler(context):
            return {}

        plugin = RetryPlugin(handler)

        assert plugin.name == "retry"


class TestMetricsPlugin:
    """Tests for MetricsPlugin."""

    @pytest.mark.asyncio
    async def test_metrics_tracking_success(self):
        """Test metrics tracking for successful execution."""
        async def handler(context):
            await asyncio.sleep(0.01)  # 10ms
            return {"result": "success"}

        plugin = MetricsPlugin(handler)

        await plugin.execute({"input": "test"})

        assert plugin.count == 1
        assert plugin.errors == 0
        assert plugin.total_time_ms > 0
        assert plugin.success_rate == 100.0

    @pytest.mark.asyncio
    async def test_metrics_tracking_failure(self):
        """Test metrics tracking for failed execution."""
        async def handler(context):
            await asyncio.sleep(0.01)
            raise RuntimeError("Failure")

        plugin = MetricsPlugin(handler)

        with pytest.raises(RuntimeError):
            await plugin.execute({"input": "test"})

        assert plugin.count == 1
        assert plugin.errors == 1
        assert plugin.total_time_ms > 0
        assert plugin.success_rate == 0.0

    @pytest.mark.asyncio
    async def test_metrics_multiple_executions(self):
        """Test metrics across multiple executions."""
        call_count = 0

        async def handler(context):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Fail once")
            return {"result": "success"}

        plugin = MetricsPlugin(handler)

        # Execute 3 times (1 success, 1 fail, 1 success)
        await plugin.execute({"input": "1"})

        try:
            await plugin.execute({"input": "2"})
        except RuntimeError:
            pass

        await plugin.execute({"input": "3"})

        assert plugin.count == 3
        assert plugin.errors == 1
        assert plugin.success_rate == pytest.approx(66.67, rel=0.1)

    @pytest.mark.asyncio
    async def test_metrics_avg_time(self):
        """Test average time calculation."""
        async def handler(context):
            await asyncio.sleep(0.01)
            return {"result": "done"}

        plugin = MetricsPlugin(handler)

        # Execute twice
        await plugin.execute({"input": "1"})
        await plugin.execute({"input": "2"})

        assert plugin.count == 2
        assert plugin.avg_time_ms > 0
        assert plugin.avg_time_ms == plugin.total_time_ms / 2

    @pytest.mark.asyncio
    async def test_metrics_properties(self):
        """Test metrics property access."""
        async def handler(context):
            return {"result": "done"}

        plugin = MetricsPlugin(handler)

        await plugin.execute({"input": "test"})

        metrics = plugin.metrics

        assert "count" in metrics
        assert "total_time_ms" in metrics
        assert "errors" in metrics
        assert "success_rate" in metrics
        assert "avg_time_ms" in metrics

    def test_metrics_plugin_name(self):
        """Test metrics plugin name."""
        async def handler(context):
            return {}

        plugin = MetricsPlugin(handler)

        assert plugin.name == "metrics"


class TestCircuitBreakerPlugin:
    """Tests for CircuitBreakerPlugin."""

    @pytest.mark.asyncio
    async def test_circuit_closed_success(self):
        """Test circuit remains closed on success."""
        async def handler(context):
            return {"result": "success"}

        plugin = CircuitBreakerPlugin(handler, failure_threshold=3)

        result = await plugin.execute({"input": "test"})

        assert result["result"] == "success"
        assert plugin.is_open is False
        assert plugin.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        async def handler(context):
            raise RuntimeError("Failure")

        plugin = CircuitBreakerPlugin(
            handler,
            failure_threshold=3,
            reset_timeout_seconds=60,
        )

        # Fail 3 times to reach threshold
        for i in range(3):
            with pytest.raises(RuntimeError):
                await plugin.execute({"input": f"test_{i}"})

        assert plugin.is_open is True
        assert plugin.failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_open_fails_immediately(self):
        """Test open circuit fails immediately."""
        async def handler(context):
            raise RuntimeError("Failure")

        plugin = CircuitBreakerPlugin(
            handler,
            failure_threshold=2,
            reset_timeout_seconds=60,
        )

        # Open the circuit
        for i in range(2):
            with pytest.raises(RuntimeError):
                await plugin.execute({"input": f"test_{i}"})

        assert plugin.is_open is True

        # Next call should fail immediately with RuntimeError
        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            await plugin.execute({"input": "test_blocked"})

    @pytest.mark.asyncio
    async def test_circuit_reset_after_timeout(self):
        """Test circuit attempts reset after timeout."""
        call_count = 0

        async def handler(context):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Failure")
            return {"result": "success"}

        plugin = CircuitBreakerPlugin(
            handler,
            failure_threshold=2,
            reset_timeout_seconds=0.1,  # 100ms timeout
        )

        # Open the circuit
        for i in range(2):
            with pytest.raises(RuntimeError):
                await plugin.execute({"input": f"test_{i}"})

        assert plugin.is_open is True

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Should attempt reset and succeed
        result = await plugin.execute({"input": "test_reset"})

        assert result["result"] == "success"
        assert plugin.is_open is False
        assert plugin.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_reset_on_success(self):
        """Test circuit resets on successful execution."""
        call_count = 0

        async def handler(context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Failure")
            return {"result": "success"}

        plugin = CircuitBreakerPlugin(handler, failure_threshold=5)

        # Fail once
        with pytest.raises(RuntimeError):
            await plugin.execute({"input": "test_1"})

        assert plugin.failure_count == 1

        # Succeed - should reset
        result = await plugin.execute({"input": "test_2"})

        assert result["result"] == "success"
        assert plugin.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_incremental_failures(self):
        """Test circuit tracks incremental failures."""
        failure_count = 0

        async def handler(context):
            nonlocal failure_count
            failure_count += 1
            raise RuntimeError(f"Failure {failure_count}")

        plugin = CircuitBreakerPlugin(handler, failure_threshold=3)

        # First failure
        with pytest.raises(RuntimeError):
            await plugin.execute({"input": "test_1"})
        assert plugin.failure_count == 1
        assert plugin.is_open is False

        # Second failure
        with pytest.raises(RuntimeError):
            await plugin.execute({"input": "test_2"})
        assert plugin.failure_count == 2
        assert plugin.is_open is False

        # Third failure - should open circuit
        with pytest.raises(RuntimeError):
            await plugin.execute({"input": "test_3"})
        assert plugin.failure_count == 3
        assert plugin.is_open is True

    def test_circuit_breaker_plugin_name(self):
        """Test circuit breaker plugin name."""
        async def handler(context):
            return {}

        plugin = CircuitBreakerPlugin(handler)

        assert plugin.name == "circuit_breaker"


class TestPluginFactory:
    """Tests for PluginFactory."""

    @pytest.mark.asyncio
    async def test_wrap_with_cache(self):
        """Test wrapping handler with cache plugin."""
        call_count = 0

        async def handler(context):
            nonlocal call_count
            call_count += 1
            return {"result": f"call_{call_count}"}

        wrapped = await PluginFactory.wrap_handler(
            handler,
            plugins=["cache"],
            config={"cache": {"ttl_seconds": 60}},
        )

        context = {"input": "test"}

        # First call
        result1 = await wrapped(context)
        assert result1["result"] == "call_1"

        # Second call - should be cached
        result2 = await wrapped(context)
        assert result2["result"] == "call_1"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_wrap_with_retry(self):
        """Test wrapping handler with retry plugin."""
        attempt_count = 0

        async def handler(context):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise RuntimeError("Fail")
            return {"result": "success"}

        wrapped = await PluginFactory.wrap_handler(
            handler,
            plugins=["retry"],
            config={"retry": {"max_retries": 3, "delay_ms": 10}},
        )

        result = await wrapped({"input": "test"})

        assert result["result"] == "success"
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_wrap_with_metrics(self):
        """Test wrapping handler with metrics plugin."""
        async def handler(context):
            return {"result": "done"}

        wrapped = await PluginFactory.wrap_handler(
            handler,
            plugins=["metrics"],
            config={},
        )

        await wrapped({"input": "test"})

        # Metrics are tracked internally in the plugin
        # Can't directly access them through wrapped handler
        # But no errors should occur

    @pytest.mark.asyncio
    async def test_wrap_with_circuit_breaker(self):
        """Test wrapping handler with circuit breaker plugin."""
        call_count = 0

        async def handler(context):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Fail")
            return {"result": "success"}

        wrapped = await PluginFactory.wrap_handler(
            handler,
            plugins=["circuit_breaker"],
            config={
                "circuit_breaker": {
                    "failure_threshold": 2,
                    "reset_timeout_seconds": 0.1,
                }
            },
        )

        # Open the circuit
        for i in range(2):
            with pytest.raises(RuntimeError):
                await wrapped({"input": f"test_{i}"})

        # Circuit should be open
        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            await wrapped({"input": "test_blocked"})

    @pytest.mark.asyncio
    async def test_wrap_with_multiple_plugins(self):
        """Test wrapping handler with multiple plugins."""
        call_count = 0

        async def handler(context):
            nonlocal call_count
            call_count += 1
            return {"result": f"call_{call_count}"}

        wrapped = await PluginFactory.wrap_handler(
            handler,
            plugins=["cache", "metrics"],
            config={"cache": {"ttl_seconds": 60}},
        )

        # Execute twice
        result1 = await wrapped({"input": "test"})
        result2 = await wrapped({"input": "test"})

        # Should be cached
        assert result1["result"] == result2["result"]
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_wrap_plugin_order(self):
        """Test plugin application order."""
        # Plugins are applied in reverse order
        # So cache -> retry means retry wraps handler, then cache wraps retry

        attempt_count = 0

        async def handler(context):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise RuntimeError("Fail first time")
            return {"result": "success"}

        wrapped = await PluginFactory.wrap_handler(
            handler,
            plugins=["cache", "retry"],
            config={
                "cache": {"ttl_seconds": 60},
                "retry": {"max_retries": 2, "delay_ms": 10},
            },
        )

        # First call - retry should handle the failure
        result1 = await wrapped({"input": "test"})
        assert result1["result"] == "success"
        assert attempt_count == 2  # Initial + 1 retry

        # Second call - cache should return cached result
        result2 = await wrapped({"input": "test"})
        assert result2["result"] == "success"
        assert attempt_count == 2  # No additional calls

    @pytest.mark.asyncio
    async def test_wrap_invalid_plugin(self):
        """Test wrapping with invalid plugin name raises error."""
        async def handler(context):
            return {"result": "done"}

        with pytest.raises(ValueError, match="Unknown plugin"):
            await PluginFactory.wrap_handler(
                handler,
                plugins=["invalid_plugin"],
                config={},
            )
