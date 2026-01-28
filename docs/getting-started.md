# Getting Started

## Installation

### Basic Installation

```bash
pip install rotalabs-cascade
```

### With Optional Dependencies

```bash
# Structured logging with structlog
pip install rotalabs-cascade[structlog]

# OpenTelemetry observability
pip install rotalabs-cascade[observability]

# Development tools (pytest, black, ruff)
pip install rotalabs-cascade[dev]

# All optional dependencies
pip install rotalabs-cascade[all]
```

## Core Dependencies

The base package requires:

- `pyyaml>=6.0`
- `python>=3.9`

## Basic Usage

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
    # Simulate ML model inference
    await asyncio.sleep(0.3)

    return {
        "result": "pass",
        "confidence": 0.85,
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

    # Execute cascade with input data
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

## Using the Event + Context Pattern

For domain-agnostic processing, use the structured Event + Context pattern:

```python
from datetime import datetime
from rotalabs_cascade import (
    CascadeEngine,
    CascadeConfig,
    UniversalEvent,
    EventContext,
    EventWithContext,
    DomainType,
    SessionContext,
    DeviceContext,
    LocationContext,
    HistoricalContext,
)

# Create a universal event (works for any domain)
event = UniversalEvent(
    id="txn_123",
    domain=DomainType.FINANCE,
    event_type="transaction",
    timestamp=datetime.now(),
    primary_entity="user_alice",      # who initiated
    secondary_entity="merchant_xyz",  # target/recipient
    value=250.00,                     # amount
    unit="USD",
    domain_data={"card_type": "credit", "merchant_category": "retail"}
)

# Create structured context
context = EventContext(
    session=SessionContext(
        ip_address="192.168.1.100",
        is_authenticated=True,
        auth_method="mfa"
    ),
    device=DeviceContext(
        device_type="mobile",
        is_trusted_device=True
    ),
    location=LocationContext(
        country="US",
        city="San Francisco",
        vpn_detected=False
    ),
    historical=HistoricalContext(
        account_age_days=730,
        previous_events_count=500,
        trust_score=0.92
    )
)

# Combine event and context
event_with_context = EventWithContext(event=event, context=context)

# Execute cascade - same logic works for ANY domain
result = await engine.execute(event_with_context.to_flat_dict())
```

## Running the Cascade

### From a Script

```python
import asyncio
from rotalabs_cascade import CascadeConfig, CascadeEngine

async def run():
    config = CascadeConfig.from_file("cascade_config.yaml")
    engine = CascadeEngine(config)

    # Register handlers...

    result = await engine.execute({"user_id": "test_user"})
    return result

result = asyncio.run(run())
```

### In an Async Web Framework

```python
from fastapi import FastAPI
from rotalabs_cascade import CascadeConfig, CascadeEngine

app = FastAPI()

# Initialize engine at startup
config = CascadeConfig.from_file("cascade_config.yaml")
engine = CascadeEngine(config)
# Register handlers...

@app.post("/evaluate")
async def evaluate(data: dict):
    result = await engine.execute(data)
    return result
```

## Next Steps

- Read [Event + Context](event-context.md) to understand the domain-agnostic input model
- Follow [Basic Cascade Tutorial](tutorials/basic-cascade.md) for a detailed walkthrough
- Learn about [APLS Learning](tutorials/apls-learning.md) for cost optimization
- See [API Reference](api/core.md) for full documentation
