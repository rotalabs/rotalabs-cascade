"""
rotalabs-cascade - Configuration-driven orchestration engine for multi-stage decision routing.

Routes decisions through configurable processing stages with dynamic routing,
parallel execution, and comprehensive plugin support.

https://rotalabs.ai
"""

__version__ = "0.1.0"

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
from rotalabs_cascade.evaluation.evaluator import ConditionEvaluator
from rotalabs_cascade.optimization.optimizer import ExecutionOptimizer
from rotalabs_cascade.plugins.builtin import (
    CachePlugin,
    CircuitBreakerPlugin,
    MetricsPlugin,
    PluginFactory,
    PluginRegistry,
    RetryPlugin,
    StagePlugin,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "CascadeEngine",
    "CascadeConfig",
    "StageConfig",
    "RoutingRule",
    "RoutingAction",
    "Condition",
    "ConditionOperator",
    # Context
    "ExecutionContext",
    "StageResult",
    # Evaluation
    "ConditionEvaluator",
    # Optimization
    "ExecutionOptimizer",
    # Plugins
    "StagePlugin",
    "PluginRegistry",
    "PluginFactory",
    "CachePlugin",
    "RetryPlugin",
    "MetricsPlugin",
    "CircuitBreakerPlugin",
]
