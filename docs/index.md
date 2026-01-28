# rotalabs-cascade

Domain-agnostic orchestration engine for multi-stage decision routing.

## What is rotalabs-cascade?

rotalabs-cascade provides a configuration-driven framework for orchestrating multi-stage decision workflows with sophisticated routing logic. It enables you to define complex processing pipelines declaratively using YAML or JSON, with runtime control over stage execution based on intermediate results.

- **Configuration-driven orchestration** - Define entire pipelines in YAML/JSON without writing orchestration code
- **Dynamic routing** - Conditionally enable, disable, or skip stages based on intermediate results
- **Domain-agnostic** - Same cascade logic works for finance, healthcare, content moderation, security, and more
- **Production-ready** - Includes caching, retries, circuit breakers, metrics, and hot-reload support

## Key Features

### Event + Context Pattern

Structured input model that separates "what happened" (the event) from "the circumstances" (context). This abstraction allows the same cascade routing logic to work across any domain.

### 5-Level Cascade Architecture

Organize processing into levels of increasing cost and sophistication:

| Level | Stage | Relative Cost | Use Case |
|-------|-------|---------------|----------|
| 1 | RULES | 1x | Simple threshold checks |
| 2 | STATISTICAL_ML | 5x | Feature-based ML models |
| 3 | SINGLE_AI | 25x | Single LLM call |
| 4 | POD | 100x | Multi-agent consensus |
| 5 | ADVERSARIAL | 500x | Adversarial validation |

### APLS (Adaptive Pattern Learning System)

Learn from execution patterns to optimize cascade routing. APLS identifies costly processing paths and generates rules to move decisions to cheaper stages, reducing compute costs while maintaining accuracy.

### Plugin System

Built-in plugins for cross-cutting concerns:

- **CachePlugin** - Cache stage results with configurable TTL
- **RetryPlugin** - Retry with exponential backoff
- **MetricsPlugin** - Collect execution metrics
- **CircuitBreakerPlugin** - Prevent cascading failures

## Supported Domains

| Domain | Use Case | Event Example |
|--------|----------|---------------|
| **Finance** | Fraud detection, transaction approval | Payments, transfers, withdrawals |
| **Healthcare** | Claims processing, triage | Insurance claims, prescriptions |
| **Content** | Moderation, spam detection | Posts, comments, uploads |
| **Security** | Access control, threat detection | Logins, API calls, data transfers |
| **Support** | Ticket routing, priority escalation | Customer tickets, complaints |
| **HR** | Resume screening, application review | Job applications, candidates |

## Package Overview

```
rotalabs_cascade/
├── core/               # Core engine and configuration
│   ├── engine          # CascadeEngine orchestration
│   ├── config          # CascadeConfig, StageConfig, RoutingRule
│   ├── context         # ExecutionContext, StageResult
│   └── event           # UniversalEvent, EventContext, EventWithContext
├── evaluation/         # Condition evaluation
│   └── evaluator       # ConditionEvaluator
├── optimization/       # Execution optimization
│   └── optimizer       # ExecutionOptimizer
├── plugins/            # Built-in plugins
│   └── builtin         # Cache, Retry, Metrics, CircuitBreaker
└── learning/           # APLS - Adaptive Pattern Learning
    ├── pattern_extractor   # Extract patterns from failures
    ├── rule_generator      # Generate rules from patterns
    ├── cost_analyzer       # Calculate migration ROI
    └── proposal            # Human-in-the-loop workflow
```

## Quick Links

- [Getting Started](getting-started.md) - Installation and first steps
- [Event + Context](event-context.md) - Understanding the domain-agnostic input model
- [Basic Cascade Tutorial](tutorials/basic-cascade.md) - Build your first cascade
- [APLS Learning Tutorial](tutorials/apls-learning.md) - Optimize routing with pattern learning
- [API Reference](api/core.md) - Detailed API documentation
