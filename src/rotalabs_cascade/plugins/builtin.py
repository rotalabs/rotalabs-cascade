"""Built-in plugins for rotalabs-cascade.

This module provides core plugin implementations including caching, retry logic,
metrics collection, circuit breaking, and a plugin registry system.

Author: Subhadip Mitra <subhadip@rotalabs.ai>
Organization: Rotalabs
"""

import hashlib
import importlib
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class StagePlugin(ABC):
    """Abstract base class for stage plugins.

    Plugins wrap stage handlers to provide additional functionality like
    caching, retries, metrics collection, or circuit breaking.

    All plugin execute methods are async to support async handlers.
    """

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plugin's logic.

        Args:
            context: Execution context containing input data and metadata

        Returns:
            Result dictionary from the plugin execution

        Raises:
            Exception: Any errors during plugin execution
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the plugin name.

        Returns:
            Unique identifier for this plugin
        """
        pass

    @property
    def version(self) -> str:
        """Get the plugin version.

        Returns:
            Version string (defaults to "1.0.0")
        """
        return "1.0.0"


class PluginRegistry:
    """Registry for managing plugins and handlers.

    The registry allows dynamic loading and retrieval of plugins and handlers,
    supporting both static registration and dynamic module loading.

    Attributes:
        _plugins: Dictionary mapping plugin names to plugin instances
        _handlers: Dictionary mapping handler names to handler functions
    """

    def __init__(self):
        """Initialize an empty plugin registry."""
        self._plugins: Dict[str, StagePlugin] = {}
        self._handlers: Dict[str, Callable] = {}
        logger.debug("Initialized PluginRegistry")

    def register_plugin(self, plugin: StagePlugin) -> None:
        """Register a plugin instance.

        Args:
            plugin: Plugin instance to register

        Raises:
            ValueError: If plugin name is already registered
        """
        name = plugin.name
        if name in self._plugins:
            logger.warning(f"Plugin '{name}' already registered, overwriting")

        self._plugins[name] = plugin
        logger.info(f"Registered plugin: {name} (version {plugin.version})")

    def register_handler(self, name: str, handler: Callable) -> None:
        """Register a handler function.

        Args:
            name: Unique identifier for the handler
            handler: Callable handler function (can be sync or async)

        Raises:
            ValueError: If handler name is already registered
        """
        if name in self._handlers:
            logger.warning(f"Handler '{name}' already registered, overwriting")

        self._handlers[name] = handler
        logger.info(f"Registered handler: {name}")

    def load_plugin_module(self, module_path: str, class_name: str) -> StagePlugin:
        """Dynamically load a plugin from a module.

        Args:
            module_path: Python module path (e.g., "my_package.plugins")
            class_name: Name of the plugin class to instantiate

        Returns:
            Instantiated plugin instance

        Raises:
            ImportError: If module cannot be imported
            AttributeError: If class not found in module
            TypeError: If class is not a StagePlugin subclass
        """
        logger.debug(f"Loading plugin: {class_name} from {module_path}")

        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)

            if not issubclass(plugin_class, StagePlugin):
                raise TypeError(f"{class_name} is not a StagePlugin subclass")

            plugin = plugin_class()
            self.register_plugin(plugin)

            logger.info(f"Successfully loaded plugin: {plugin.name}")
            return plugin

        except ImportError as e:
            logger.error(f"Failed to import module {module_path}: {e}")
            raise
        except AttributeError as e:
            logger.error(f"Class {class_name} not found in {module_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading plugin {class_name}: {e}")
            raise

    def get_handler(self, name: str) -> Optional[Callable]:
        """Get a registered handler by name.

        Args:
            name: Handler identifier

        Returns:
            Handler function or None if not found
        """
        return self._handlers.get(name)

    def get_plugin(self, name: str) -> Optional[StagePlugin]:
        """Get a registered plugin by name.

        Args:
            name: Plugin identifier

        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict[str, str]]:
        """List all registered plugins.

        Returns:
            List of dictionaries containing plugin name and version
        """
        return [
            {"name": plugin.name, "version": plugin.version}
            for plugin in self._plugins.values()
        ]


class CachePlugin(StagePlugin):
    """Plugin for caching stage execution results.

    Caches results based on input context hash with configurable TTL.
    Cache entries expire after ttl_seconds.

    Attributes:
        wrapped_handler: The underlying handler to cache
        ttl_seconds: Time-to-live for cache entries in seconds
    """

    def __init__(self, wrapped_handler: Callable, ttl_seconds: int = 300):
        """Initialize cache plugin.

        Args:
            wrapped_handler: Handler function to wrap with caching
            ttl_seconds: Cache entry TTL in seconds (default: 300)
        """
        self.wrapped_handler = wrapped_handler
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple[Dict[str, Any], float]] = {}
        logger.debug(f"Initialized CachePlugin with TTL={ttl_seconds}s")

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "cache"

    def _get_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key from context.

        Args:
            context: Execution context

        Returns:
            SHA256 hash of serialized context
        """
        # Sort keys for consistent hashing
        serialized = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _is_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid.

        Args:
            timestamp: Entry creation timestamp

        Returns:
            True if entry is within TTL, False otherwise
        """
        age = time.time() - timestamp
        return age < self.ttl_seconds

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with caching.

        Checks cache for existing result. If cache miss or expired,
        executes wrapped handler and stores result.

        Args:
            context: Execution context

        Returns:
            Cached or newly computed result

        Raises:
            Exception: Any errors from wrapped handler
        """
        cache_key = self._get_cache_key(context)

        # Check cache
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if self._is_valid(timestamp):
                logger.debug(f"Cache hit: {cache_key[:8]}...")
                return result
            else:
                logger.debug(f"Cache expired: {cache_key[:8]}...")
                del self._cache[cache_key]

        # Cache miss - execute handler
        logger.debug(f"Cache miss: {cache_key[:8]}...")

        # Support both sync and async handlers
        if hasattr(self.wrapped_handler, "__call__"):
            import inspect
            if inspect.iscoroutinefunction(self.wrapped_handler):
                result = await self.wrapped_handler(context)
            else:
                result = self.wrapped_handler(context)
        else:
            result = self.wrapped_handler(context)

        # Store in cache
        self._cache[cache_key] = (result, time.time())
        logger.debug(f"Cached result: {cache_key[:8]}...")

        return result


class RetryPlugin(StagePlugin):
    """Plugin for retry logic with exponential backoff.

    Retries failed executions with exponential backoff between attempts.

    Attributes:
        wrapped_handler: The underlying handler to retry
        max_retries: Maximum number of retry attempts
        delay_ms: Initial delay between retries in milliseconds
    """

    def __init__(
        self,
        wrapped_handler: Callable,
        max_retries: int = 3,
        delay_ms: int = 100,
    ):
        """Initialize retry plugin.

        Args:
            wrapped_handler: Handler function to wrap with retries
            max_retries: Maximum retry attempts (default: 3)
            delay_ms: Initial delay in milliseconds (default: 100)
        """
        self.wrapped_handler = wrapped_handler
        self.max_retries = max_retries
        self.delay_ms = delay_ms
        logger.debug(
            f"Initialized RetryPlugin with max_retries={max_retries}, "
            f"delay_ms={delay_ms}"
        )

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "retry"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with retry logic.

        Attempts execution up to max_retries times with exponential backoff.
        Delay doubles after each failed attempt.

        Args:
            context: Execution context

        Returns:
            Result from successful execution

        Raises:
            Exception: Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retries + 1}")

                # Support both sync and async handlers
                import inspect
                if inspect.iscoroutinefunction(self.wrapped_handler):
                    result = await self.wrapped_handler(context)
                else:
                    result = self.wrapped_handler(context)

                if attempt > 0:
                    logger.info(f"Succeeded on attempt {attempt + 1}")

                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}"
                )

                if attempt < self.max_retries:
                    # Exponential backoff
                    delay_seconds = (self.delay_ms * (2 ** attempt)) / 1000.0
                    logger.debug(f"Retrying in {delay_seconds:.2f}s...")

                    import asyncio
                    await asyncio.sleep(delay_seconds)

        # All retries exhausted
        logger.error(
            f"All {self.max_retries + 1} attempts failed. "
            f"Last error: {type(last_exception).__name__}"
        )
        raise last_exception


class MetricsPlugin(StagePlugin):
    """Plugin for collecting execution metrics.

    Tracks execution count, total time, errors, and computes success rate
    and average execution time.

    Attributes:
        wrapped_handler: The underlying handler to monitor
        count: Total number of executions
        total_time_ms: Total execution time in milliseconds
        errors: Number of failed executions
    """

    def __init__(self, wrapped_handler: Callable):
        """Initialize metrics plugin.

        Args:
            wrapped_handler: Handler function to wrap with metrics
        """
        self.wrapped_handler = wrapped_handler
        self.count = 0
        self.total_time_ms = 0.0
        self.errors = 0
        logger.debug("Initialized MetricsPlugin")

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "metrics"

    @property
    def success_rate(self) -> float:
        """Calculate success rate.

        Returns:
            Success rate as percentage (0-100)
        """
        if self.count == 0:
            return 0.0
        return ((self.count - self.errors) / self.count) * 100

    @property
    def avg_time_ms(self) -> float:
        """Calculate average execution time.

        Returns:
            Average time in milliseconds
        """
        if self.count == 0:
            return 0.0
        return self.total_time_ms / self.count

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary containing all metrics
        """
        return {
            "count": self.count,
            "total_time_ms": self.total_time_ms,
            "errors": self.errors,
            "success_rate": self.success_rate,
            "avg_time_ms": self.avg_time_ms,
        }

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with metrics collection.

        Times execution and updates metrics counters.

        Args:
            context: Execution context

        Returns:
            Result from wrapped handler

        Raises:
            Exception: Any errors from wrapped handler (recorded in metrics)
        """
        start_time = time.time()
        self.count += 1

        try:
            # Support both sync and async handlers
            import inspect
            if inspect.iscoroutinefunction(self.wrapped_handler):
                result = await self.wrapped_handler(context)
            else:
                result = self.wrapped_handler(context)

            elapsed_ms = (time.time() - start_time) * 1000
            self.total_time_ms += elapsed_ms

            logger.debug(
                f"Execution completed in {elapsed_ms:.2f}ms "
                f"(avg: {self.avg_time_ms:.2f}ms, "
                f"success rate: {self.success_rate:.1f}%)"
            )

            return result

        except Exception as e:
            self.errors += 1
            elapsed_ms = (time.time() - start_time) * 1000
            self.total_time_ms += elapsed_ms

            logger.error(
                f"Execution failed after {elapsed_ms:.2f}ms: "
                f"{type(e).__name__}: {e} "
                f"(success rate: {self.success_rate:.1f}%)"
            )
            raise


class CircuitBreakerPlugin(StagePlugin):
    """Plugin implementing circuit breaker pattern.

    Prevents cascading failures by opening circuit after threshold failures.
    Circuit automatically resets after timeout period.

    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Circuit tripped, requests fail immediately
        - HALF_OPEN: Testing if service recovered (auto after timeout)

    Attributes:
        wrapped_handler: The underlying handler to protect
        failure_threshold: Number of failures before opening circuit
        reset_timeout_seconds: Time before attempting reset
        failure_count: Current consecutive failure count
        last_failure_time: Timestamp of last failure
        is_open: Whether circuit is currently open
    """

    def __init__(
        self,
        wrapped_handler: Callable,
        failure_threshold: int = 5,
        reset_timeout_seconds: int = 60,
    ):
        """Initialize circuit breaker plugin.

        Args:
            wrapped_handler: Handler function to protect
            failure_threshold: Failures before opening (default: 5)
            reset_timeout_seconds: Timeout before reset attempt (default: 60)
        """
        self.wrapped_handler = wrapped_handler
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False
        logger.debug(
            f"Initialized CircuitBreakerPlugin with "
            f"threshold={failure_threshold}, timeout={reset_timeout_seconds}s"
        )

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "circuit_breaker"

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset.

        Returns:
            True if timeout period has elapsed since last failure
        """
        if self.last_failure_time is None:
            return False

        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.reset_timeout_seconds

    def _record_success(self) -> None:
        """Record successful execution and reset circuit."""
        if self.is_open:
            logger.info("Circuit breaker reset after successful execution")

        self.failure_count = 0
        self.is_open = False
        self.last_failure_time = None

    def _record_failure(self) -> None:
        """Record failed execution and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold and not self.is_open:
            self.is_open = True
            logger.error(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with circuit breaker protection.

        Checks circuit state before execution. If open, fails immediately
        unless timeout has passed (half-open state).

        Args:
            context: Execution context

        Returns:
            Result from successful execution

        Raises:
            RuntimeError: If circuit is open
            Exception: Any errors from wrapped handler
        """
        # Check if circuit is open
        if self.is_open:
            if self._should_attempt_reset():
                logger.info("Circuit breaker attempting reset (half-open)")
            else:
                error_msg = (
                    f"Circuit breaker is OPEN (failures: {self.failure_count}). "
                    f"Reset in {self.reset_timeout_seconds - (time.time() - self.last_failure_time):.1f}s"
                )
                logger.warning(error_msg)
                raise RuntimeError(error_msg)

        try:
            # Support both sync and async handlers
            import inspect
            if inspect.iscoroutinefunction(self.wrapped_handler):
                result = await self.wrapped_handler(context)
            else:
                result = self.wrapped_handler(context)

            self._record_success()
            return result

        except Exception as e:
            self._record_failure()
            logger.error(
                f"Circuit breaker recorded failure {self.failure_count}/"
                f"{self.failure_threshold}: {type(e).__name__}: {e}"
            )
            raise


class PluginFactory:
    """Factory for composing plugins around handlers.

    Provides utilities for wrapping handlers with multiple plugins in a
    composable manner. Plugins are applied in reverse order so that the
    first plugin in the list is the outermost wrapper.
    """

    @staticmethod
    async def wrap_handler(
        handler: Callable,
        plugins: List[str],
        config: Dict[str, Any],
    ) -> Callable:
        """Wrap a handler with multiple plugins.

        Applies plugins in reverse order to create a plugin chain:
        cache -> retry -> metrics -> circuit_breaker -> handler

        Args:
            handler: Base handler to wrap
            plugins: List of plugin names to apply (e.g., ["cache", "retry"])
            config: Configuration dict with plugin-specific settings
                   Example: {
                       "cache": {"ttl_seconds": 600},
                       "retry": {"max_retries": 5},
                       "circuit_breaker": {"failure_threshold": 3}
                   }

        Returns:
            Wrapped handler with all plugins applied

        Raises:
            ValueError: If unknown plugin name provided
        """
        wrapped = handler

        # Apply plugins in reverse order (innermost first)
        for plugin_name in reversed(plugins):
            plugin_config = config.get(plugin_name, {})

            if plugin_name == "cache":
                ttl = plugin_config.get("ttl_seconds", 300)
                plugin = CachePlugin(wrapped, ttl_seconds=ttl)
                logger.debug(f"Applied CachePlugin with TTL={ttl}s")

            elif plugin_name == "retry":
                max_retries = plugin_config.get("max_retries", 3)
                delay_ms = plugin_config.get("delay_ms", 100)
                plugin = RetryPlugin(
                    wrapped, max_retries=max_retries, delay_ms=delay_ms
                )
                logger.debug(
                    f"Applied RetryPlugin with max_retries={max_retries}"
                )

            elif plugin_name == "metrics":
                plugin = MetricsPlugin(wrapped)
                logger.debug("Applied MetricsPlugin")

            elif plugin_name == "circuit_breaker":
                threshold = plugin_config.get("failure_threshold", 5)
                timeout = plugin_config.get("reset_timeout_seconds", 60)
                plugin = CircuitBreakerPlugin(
                    wrapped, failure_threshold=threshold, reset_timeout_seconds=timeout
                )
                logger.debug(
                    f"Applied CircuitBreakerPlugin with threshold={threshold}"
                )

            else:
                raise ValueError(f"Unknown plugin: {plugin_name}")

            wrapped = plugin.execute

        logger.info(f"Created handler with plugins: {' -> '.join(plugins)}")
        return wrapped
