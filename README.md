# rotalabs-cascade

Configuration-driven orchestration engine for multi-stage decision routing with dynamic execution control and comprehensive plugin support.

## Overview

**rotalabs-cascade** provides a powerful framework for orchestrating multi-stage decision workflows with sophisticated routing logic. It enables you to define complex processing pipelines declaratively using YAML or JSON, with runtime control over stage execution based on intermediate results.

### Key Value Proposition

- **Dynamic routing**: Conditionally enable, disable, or skip stages based on intermediate results
- **Rich condition language**: 21 operators including comparison, logical, collection, pattern matching, and statistical operations
- **Async-native**: Built from the ground up for async/await with parallel execution support
- **Configuration-driven**: Define entire pipelines in YAML/JSON without writing orchestration code
- **Production-ready**: Includes caching, retries, circuit breakers, metrics, and hot-reload

### Use Cases

- **Approval workflows**: Route requests through different approval chains based on risk scores
- **Policy evaluation**: Apply policies in stages with conditional escalation
- **Trust scoring**: Cascade through fast/medium/expensive checks based on confidence thresholds
- **Resource allocation**: Dynamically route to different resource pools based on load and priority
- **Content moderation**: Multi-stage filtering with early termination or escalation

## Key Features

- **Configuration-driven** - Define cascades in YAML/JSON, no orchestration code needed
- **Async-native execution** - Full async/await support with `asyncio.timeout`
- **Dynamic routing** - Enable/disable stages, skip ahead, or terminate based on stage results
- **Parallel execution** - Run independent stages concurrently with configurable parallelism
- **Rich condition language** - 21 operators: comparison, logical, collection, pattern, statistical
- **Plugin system** - Built-in cache, retry, metrics, and circuit breaker plugins
- **Zero-copy data passing** - Efficient context sharing across stages
- **Hot-reload configuration** - Update pipeline definitions without restart
- **Dependency resolution** - Automatic topological sort based on stage dependencies
- **Comprehensive observability** - Timeline tracking, routing decisions, and stage metrics

## Installation

```bash
# Basic installation
pip install rotalabs-cascade

# With optional dependencies
pip install rotalabs-cascade[structlog]      # Structured logging
pip install rotalabs-cascade[observability]  # OpenTelemetry support
pip install rotalabs-cascade[dev]            # Development tools (pytest, black, ruff)
pip install rotalabs-cascade[all]            # All optional dependencies
```

## Quick Start

### 1. Define Your Cascade Configuration

Create a YAML configuration file defining your stages and routing rules:

```yaml
# cascade_config.yaml
name: trust_scoring_cascade
version: "1.0"

stages:
  FAST_CHECK:
    name: FAST_CHECK
    enabled: true
    timeout_ms: 100
    routing_rules:
      - name: low_confidence_escalate
        type: routing
        priority: 100
        condition:
          field: stages.FAST_CHECK.confidence
          operator: "<"
          value: 0.8
        action:
          type: enable_stages
          stages: ["MEDIUM_CHECK"]

      - name: high_confidence_terminate
        type: routing
        priority: 90
        condition:
          field: stages.FAST_CHECK.confidence
          operator: ">="
          value: 0.95
        action:
          type: terminate

  MEDIUM_CHECK:
    name: MEDIUM_CHECK
    enabled: false  # Only enabled if FAST_CHECK triggers it
    timeout_ms: 500
    depends_on: ["FAST_CHECK"]
    routing_rules:
      - name: still_uncertain
        type: routing
        priority: 100
        condition:
          field: stages.MEDIUM_CHECK.confidence
          operator: "<"
          value: 0.9
        action:
          type: enable_stages
          stages: ["EXPENSIVE_CHECK"]

  EXPENSIVE_CHECK:
    name: EXPENSIVE_CHECK
    enabled: false
    timeout_ms: 2000
    depends_on: ["MEDIUM_CHECK"]

execution_order:
  - FAST_CHECK
  - MEDIUM_CHECK
  - EXPENSIVE_CHECK

global_timeout_ms: 5000
max_parallel_stages: 3
```

### 2. Implement Stage Handlers

Create async handler functions for each stage:

```python
import asyncio
from rotalabs_cascade import CascadeConfig, CascadeEngine, ExecutionContext

# Implement your stage handlers
async def fast_check_handler(context: ExecutionContext) -> dict:
    """Fast heuristic check."""
    user_id = context.get("user_id")

    # Quick validation logic
    confidence = 0.75 if user_id.startswith("trusted_") else 0.4

    return {
        "result": "pass" if confidence > 0.5 else "review",
        "confidence": confidence,
        "data": {"method": "heuristic", "checks": ["basic_validation"]}
    }

async def medium_check_handler(context: ExecutionContext) -> dict:
    """Medium complexity ML model check."""
    user_id = context.get("user_id")

    # Simulate ML model inference
    await asyncio.sleep(0.3)
    confidence = 0.85

    return {
        "result": "pass",
        "confidence": confidence,
        "data": {"method": "ml_model", "model": "trust_v2"}
    }

async def expensive_check_handler(context: ExecutionContext) -> dict:
    """Expensive deep analysis."""
    # Simulate expensive computation
    await asyncio.sleep(1.5)

    return {
        "result": "pass",
        "confidence": 0.99,
        "data": {"method": "deep_analysis", "checks_run": 47}
    }
```

### 3. Execute the Cascade

Load configuration, register handlers, and execute:

```python
async def main():
    # Load configuration
    config = CascadeConfig.from_file("cascade_config.yaml")

    # Create engine
    engine = CascadeEngine(config)

    # Register stage handlers
    engine.register_stage("FAST_CHECK", fast_check_handler)
    engine.register_stage("MEDIUM_CHECK", medium_check_handler)
    engine.register_stage("EXPENSIVE_CHECK", expensive_check_handler)

    # Execute cascade
    result = await engine.execute({
        "user_id": "user_12345",
        "action": "withdraw",
        "amount": 10000
    })

    # Inspect results
    print(f"Success: {result['success']}")
    print(f"Execution time: {result['execution_time_ms']:.2f}ms")
    print(f"Stages executed: {result['stages_executed']}")

    for stage_name, stage_result in result["stage_results"].items():
        print(f"\n{stage_name}:")
        print(f"  Result: {stage_result['result']}")
        print(f"  Confidence: {stage_result.get('confidence', 'N/A')}")
        print(f"  Time: {stage_result['time_ms']:.2f}ms")

if __name__ == "__main__":
    asyncio.run(main())
```

**Output:**
```
Success: True
Execution time: 345.67ms
Stages executed: 2

FAST_CHECK:
  Result: review
  Confidence: 0.4
  Time: 2.34ms

MEDIUM_CHECK:
  Result: pass
  Confidence: 0.85
  Time: 312.45ms
```

## Configuration Reference

### StageConfig

Complete configuration options for a stage:

```python
StageConfig(
    name="STAGE_NAME",                    # Unique stage identifier
    enabled=True,                         # Initial enabled state
    handler_type="custom",                # Handler type (optional)
    timeout_ms=30000,                     # Stage execution timeout
    max_retries=0,                        # Number of retry attempts
    retry_delay_ms=1000,                  # Delay between retries
    can_run_parallel=False,               # Allow parallel execution
    parallel_group="group_a",             # Parallel group identifier
    depends_on=["STAGE_1", "STAGE_2"],    # Stage dependencies
    routing_rules=[...],                  # Routing rules (see below)
    cache_enabled=False,                  # Enable result caching
    cache_ttl_seconds=3600,               # Cache TTL
    custom_properties={"key": "value"}    # Domain-specific properties
)
```

### RoutingRule

Define conditional routing behavior:

```python
RoutingRule(
    name="rule_name",                     # Unique rule identifier
    type="routing",                       # Rule type: precondition, routing, postcondition
    priority=100,                         # Execution priority (higher first)
    condition=Condition(...),             # Condition to evaluate
    action=RoutingAction(...)             # Action to execute if condition matches
)
```

**Rule types:**
- `precondition`: Evaluated before stage execution (can prevent execution)
- `routing`: Evaluated after stage execution (controls flow)
- `postcondition`: Evaluated after stage execution (cleanup/notifications)

### Condition Operators

The engine supports 21 operators for flexible condition evaluation:

| Category | Operators | Description |
|----------|-----------|-------------|
| **Comparison** | `==`, `!=`, `>`, `>=`, `<`, `<=` | Standard comparisons |
| **Logical** | `AND`, `OR`, `NOT` | Combine multiple conditions |
| **Collection** | `IN`, `NOT_IN`, `CONTAINS` | Membership testing |
| **Pattern** | `MATCHES` | Regular expression matching |
| **Existence** | `EXISTS`, `IS_NULL` | Field presence checks |
| **Aggregation** | `ALL`, `ANY`, `NONE` | List element matching |
| **Statistical** | `SUM`, `AVG`, `MIN`, `MAX`, `COUNT` | List statistics |

**Example conditions:**

```yaml
# Simple comparison
condition:
  field: stages.FAST_CHECK.confidence
  operator: "<"
  value: 0.8

# Logical AND
condition:
  operator: AND
  conditions:
    - field: user.risk_score
      operator: ">"
      value: 0.7
    - field: transaction.amount
      operator: ">="
      value: 10000

# Pattern matching
condition:
  field: user.email
  operator: MATCHES
  value: ".*@trusted-domain\\.com$"

# Collection operations
condition:
  field: user.roles
  operator: CONTAINS
  value: "admin"

# Statistical operators
condition:
  field: previous_transactions.amounts
  operator: AVG
  value: 5000
```

### Routing Actions

Execute actions when conditions match:

| Action Type | Description | Parameters |
|-------------|-------------|------------|
| `terminate` | Stop cascade execution | None |
| `skip_to` | Jump to specific stage | `target`: stage name |
| `enable_stages` | Dynamically enable stages | `stages`: list of stage names |
| `disable_stages` | Dynamically disable stages | `stages`: list of stage names |
| `set_field` | Modify context data | `field`: path, `value`: new value |

**Examples:**

```yaml
# Terminate early on high confidence
action:
  type: terminate

# Skip to expensive check
action:
  type: skip_to
  target: EXPENSIVE_CHECK

# Enable multiple stages
action:
  type: enable_stages
  stages: ["MANUAL_REVIEW", "NOTIFY_COMPLIANCE"]

# Update context
action:
  type: set_field
  field: user.risk_level
  value: high
```

## Plugins

Use built-in plugins to enhance stage handlers with cross-cutting concerns.

### Available Plugins

**CachePlugin** - Cache stage results with TTL:
```python
from rotalabs_cascade import CachePlugin

cached_handler = CachePlugin(
    wrapped_handler=my_handler,
    ttl_seconds=600  # Cache for 10 minutes
)
```

**RetryPlugin** - Retry with exponential backoff:
```python
from rotalabs_cascade import RetryPlugin

retry_handler = RetryPlugin(
    wrapped_handler=my_handler,
    max_retries=3,
    delay_ms=100  # 100ms, 200ms, 400ms delays
)
```

**MetricsPlugin** - Collect execution metrics:
```python
from rotalabs_cascade import MetricsPlugin

metrics_handler = MetricsPlugin(wrapped_handler=my_handler)

# Access metrics
print(metrics_handler.metrics)
# {'count': 42, 'total_time_ms': 1234.5, 'errors': 2,
#  'success_rate': 95.2, 'avg_time_ms': 29.4}
```

**CircuitBreakerPlugin** - Prevent cascading failures:
```python
from rotalabs_cascade import CircuitBreakerPlugin

protected_handler = CircuitBreakerPlugin(
    wrapped_handler=my_handler,
    failure_threshold=5,      # Open after 5 failures
    reset_timeout_seconds=60  # Try again after 60s
)
```

### Composing Plugins

Use `PluginFactory` to compose multiple plugins:

```python
from rotalabs_cascade import PluginFactory

# Compose multiple plugins: cache -> retry -> metrics -> circuit_breaker -> handler
wrapped = await PluginFactory.wrap_handler(
    handler=my_handler,
    plugins=["cache", "retry", "metrics", "circuit_breaker"],
    config={
        "cache": {"ttl_seconds": 600},
        "retry": {"max_retries": 5, "delay_ms": 200},
        "circuit_breaker": {"failure_threshold": 3, "reset_timeout_seconds": 30}
    }
)

engine.register_stage("MY_STAGE", wrapped)
```

## Advanced Features

### Hot-Reload Configuration

Update cascade configuration without restarting:

```python
# Load new configuration
new_config = CascadeConfig.from_file("updated_cascade.yaml")

# Hot-reload (preserves handlers and statistics)
engine.update_config(new_config)
```

### Zero-Copy Context Access

Access input data efficiently using dot notation:

```python
async def my_handler(context: ExecutionContext) -> dict:
    # Dot notation with caching
    user_name = context.get("user.profile.name")
    settings = context.get("user.settings", default={})

    # Modify context
    context.set("user.verified", True)

    # Access previous stage results
    fast_result = context.get_stage_result("FAST_CHECK")
    if fast_result and fast_result.confidence > 0.9:
        # Use fast result
        return {"result": fast_result.result}

    return {"result": "pass"}
```

### Execution Statistics

Monitor performance across all stages:

```python
stats = engine.get_statistics()

for stage_name, metrics in stats.items():
    print(f"{stage_name}:")
    print(f"  Executions: {metrics['count']}")
    print(f"  Avg time: {metrics['total_time_ms'] / metrics['count']:.2f}ms")
    print(f"  Errors: {metrics['errors']}")
```

### Cache Management

```python
# Clear all caches (result and execution plan caches)
engine.clear_cache()
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/rotalabs/rotalabs-cascade.git
cd rotalabs-cascade

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=rotalabs_cascade --cov-report=html

# Run specific test
pytest tests/test_engine.py::test_basic_execution -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

## Architecture

### Core Components

- **CascadeEngine**: Main orchestration engine managing execution
- **CascadeConfig**: Configuration schema with validation
- **ExecutionContext**: Tracks state and results throughout execution
- **ConditionEvaluator**: Evaluates routing conditions
- **ExecutionOptimizer**: Optimizes execution plans
- **PluginRegistry**: Manages plugins and handlers

### Execution Flow

1. Load and validate configuration
2. Generate execution plan (topological sort or explicit order)
3. Execute stages in order with dependency checking
4. Evaluate routing rules after each stage
5. Apply routing actions (enable/disable/skip/terminate)
6. Return comprehensive execution result

## Performance Considerations

- **Memory efficient**: Uses `__slots__` in hot-path classes
- **Zero-copy data**: Context stores reference to input data
- **Cached lookups**: Dot notation paths cached after first access
- **Compiled rules**: Routing rules organized by priority on startup
- **Plan caching**: Execution plans cached per input pattern
- **Result caching**: Stage results cached with configurable TTL

## Links

- **PyPI**: https://pypi.org/project/rotalabs-cascade/
- **GitHub**: https://github.com/rotalabs/rotalabs-cascade
- **Documentation**: https://rotalabs.github.io/rotalabs-cascade/
- **Website**: https://rotalabs.ai

## License

MIT License - see LICENSE file for details.

## Authors

- Subhadip Mitra (subhadip@rotalabs.ai)
- Rotalabs Research (research@rotalabs.ai)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
