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
from rotalabs_cascade.core.event import (
    DomainType,
    EventContext,
    EventType,
    EventWithContext,
    UniversalEvent,
    SessionContext,
    DeviceContext,
    LocationContext,
    HistoricalContext,
    EntityContext,
    create_finance_event,
    create_content_event,
    create_security_event,
    create_support_event,
)
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

# Learning module (APLS - Adaptive Pattern Learning System)
from rotalabs_cascade.learning import (
    CostAnalyzer,
    GeneratedRule,
    MigrationROI,
    PatternConfig,
    PatternExtractor,
    PatternLearningInsight,
    PatternType,
    ProposalManager,
    ProposalStatus,
    RuleGenerator,
    RuleProposal,
    RuleTemplate,
    StageCost,
    StageFailurePattern,
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
    # Event + Context (domain-agnostic)
    "DomainType",
    "EventType",
    "UniversalEvent",
    "EventContext",
    "EventWithContext",
    "SessionContext",
    "DeviceContext",
    "LocationContext",
    "HistoricalContext",
    "EntityContext",
    "create_finance_event",
    "create_content_event",
    "create_security_event",
    "create_support_event",
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
    # Learning (APLS)
    "PatternExtractor",
    "PatternConfig",
    "PatternType",
    "StageFailurePattern",
    "PatternLearningInsight",
    "RuleGenerator",
    "RuleTemplate",
    "GeneratedRule",
    "CostAnalyzer",
    "StageCost",
    "MigrationROI",
    "ProposalManager",
    "ProposalStatus",
    "RuleProposal",
]
