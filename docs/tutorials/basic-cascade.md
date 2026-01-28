# Building Your First Cascade

This tutorial walks you through building a complete 3-stage cascade system using rotalabs-cascade. You will learn how to define stages, register handlers, configure routing rules, and execute the cascade with different inputs.

## What We're Building

A typical cascade system routes events through progressively more sophisticated (and expensive) stages:

1. **RULES** - Fast, deterministic rule checks (microseconds, nearly free)
2. **ML** - Statistical/ML model inference (milliseconds, moderate cost)
3. **AI** - LLM-based analysis (seconds, high cost)

The goal is to resolve as many cases as possible in earlier stages, only escalating to more expensive stages when necessary.

## Prerequisites

Install rotalabs-cascade:

```bash
pip install rotalabs-cascade
```

## Step 1: Define the Cascade Configuration

First, create the configuration that defines your stages and routing rules:

```python
from rotalabs_cascade import (
    CascadeConfig,
    StageConfig,
    RoutingRule,
    RoutingAction,
    Condition,
    ConditionOperator,
)

# Define the three stages
stages = {
    "RULES": StageConfig(
        name="RULES",
        enabled=True,
        timeout_ms=1000,
        routing_rules=[
            # If rules stage has low confidence, enable ML
            RoutingRule(
                name="escalate_to_ml",
                type="routing",
                condition=Condition(
                    field="stages.RULES.confidence",
                    operator=ConditionOperator.LT,
                    value=0.9,
                ),
                action=RoutingAction(
                    type="enable_stages",
                    stages=["ML"],
                ),
                priority=10,
            ),
        ],
    ),
    "ML": StageConfig(
        name="ML",
        enabled=False,  # Only enabled by RULES stage
        timeout_ms=5000,
        depends_on=["RULES"],
        routing_rules=[
            # If ML has low confidence, enable AI
            RoutingRule(
                name="escalate_to_ai",
                type="routing",
                condition=Condition(
                    field="stages.ML.confidence",
                    operator=ConditionOperator.LT,
                    value=0.8,
                ),
                action=RoutingAction(
                    type="enable_stages",
                    stages=["AI"],
                ),
                priority=10,
            ),
        ],
    ),
    "AI": StageConfig(
        name="AI",
        enabled=False,  # Only enabled by ML stage
        timeout_ms=30000,
        depends_on=["ML"],
    ),
}

# Create the cascade configuration
config = CascadeConfig(
    name="my_first_cascade",
    version="1.0.0",
    stages=stages,
    execution_order=["RULES", "ML", "AI"],
    global_timeout_ms=60000,
    max_parallel_stages=1,
)
```

### Understanding the Configuration

- **StageConfig**: Defines each stage with its name, timeout, dependencies, and routing rules
- **enabled**: Stages start disabled and are enabled dynamically based on routing rules
- **depends_on**: Ensures stages execute in the correct order
- **routing_rules**: Define conditions that enable other stages or modify execution flow
- **execution_order**: The planned order of stage execution

## Step 2: Create the Cascade Engine

Initialize the engine with your configuration:

```python
from rotalabs_cascade import CascadeEngine

engine = CascadeEngine(config)
```

## Step 3: Implement Stage Handlers

Each stage needs an async handler function that processes the execution context and returns a result:

```python
from rotalabs_cascade import ExecutionContext
from typing import Dict, Any

async def rules_handler(context: ExecutionContext) -> Dict[str, Any]:
    """Fast rule-based checks.

    This handler runs deterministic rules to handle clear-cut cases.
    """
    # Get input data
    value = context.get("value", 0)
    domain = context.get("domain", "GENERIC")
    trust_score = context.get("trust_score", 0.5)

    # Apply simple rules
    if value < 10:
        # Trivial value - auto-approve with high confidence
        return {
            "result": "APPROVE",
            "confidence": 0.99,
            "data": {"reason": "below_threshold"}
        }

    if trust_score > 0.95:
        # Highly trusted entity - auto-approve
        return {
            "result": "APPROVE",
            "confidence": 0.95,
            "data": {"reason": "high_trust"}
        }

    if value > 10000:
        # Very high value - needs deeper analysis
        return {
            "result": "UNCERTAIN",
            "confidence": 0.3,
            "data": {"reason": "high_value_needs_review"}
        }

    # Default: moderate confidence, may need ML
    return {
        "result": "UNCERTAIN",
        "confidence": 0.7,
        "data": {"reason": "standard_check"}
    }


async def ml_handler(context: ExecutionContext) -> Dict[str, Any]:
    """ML model inference.

    This handler runs a trained model for pattern-based detection.
    """
    # Get features for ML model
    value = context.get("value", 0)
    account_age_days = context.get("account_age_days", 0)
    previous_events_count = context.get("historical.previous_events_count", 0)

    # Simulate ML model prediction
    # In production, this would call your actual model
    features = {
        "value_normalized": min(value / 1000, 10),
        "account_maturity": min(account_age_days / 365, 5),
        "activity_level": min(previous_events_count / 100, 10),
    }

    # Simple heuristic (replace with actual model)
    score = (
        features["account_maturity"] * 0.3 +
        features["activity_level"] * 0.3 +
        (1 - features["value_normalized"] / 10) * 0.4
    )

    if score > 0.7:
        return {
            "result": "APPROVE",
            "confidence": min(0.6 + score * 0.3, 0.95),
            "data": {"ml_score": score, "features": features}
        }
    elif score < 0.3:
        return {
            "result": "REJECT",
            "confidence": min(0.6 + (1 - score) * 0.3, 0.95),
            "data": {"ml_score": score, "features": features}
        }
    else:
        # Uncertain - may need AI analysis
        return {
            "result": "UNCERTAIN",
            "confidence": 0.5 + abs(score - 0.5) * 0.3,
            "data": {"ml_score": score, "features": features}
        }


async def ai_handler(context: ExecutionContext) -> Dict[str, Any]:
    """LLM-based deep analysis.

    This handler uses an LLM for nuanced reasoning on edge cases.
    """
    # Get all available context
    event_data = context.get("event", {})
    domain_data = event_data.get("domain_data", {})

    # Get previous stage results for context
    rules_result = context.get_stage_result("RULES")
    ml_result = context.get_stage_result("ML")

    # Build reasoning context
    reasoning_context = {
        "rules_result": rules_result.to_dict() if rules_result else None,
        "ml_result": ml_result.to_dict() if ml_result else None,
        "event_data": event_data,
    }

    # Simulate LLM analysis
    # In production, this would call your actual LLM
    # For demo, we make a decision based on accumulated evidence

    ml_score = 0.5
    if ml_result and ml_result.data:
        ml_score = ml_result.data.get("ml_score", 0.5)

    # AI makes final determination with explanation
    if ml_score > 0.5:
        return {
            "result": "APPROVE",
            "confidence": 0.92,
            "data": {
                "reasoning": "Based on account history and behavioral patterns, "
                            "this activity appears consistent with normal usage.",
                "factors": ["established_account", "consistent_behavior", "low_risk_profile"]
            }
        }
    else:
        return {
            "result": "REJECT",
            "confidence": 0.88,
            "data": {
                "reasoning": "Multiple risk indicators detected. Activity pattern "
                            "deviates significantly from expected behavior.",
                "factors": ["behavioral_anomaly", "elevated_risk", "insufficient_trust"]
            }
        }
```

## Step 4: Register Handlers with the Engine

Connect your handlers to the engine:

```python
# Register each handler with its stage name
engine.register_stage("RULES", rules_handler)
engine.register_stage("ML", ml_handler)
engine.register_stage("AI", ai_handler)
```

## Step 5: Execute the Cascade

Now run the cascade with different inputs to see how routing works:

```python
import asyncio
from datetime import datetime
from rotalabs_cascade import (
    UniversalEvent,
    EventContext,
    EventWithContext,
    DomainType,
    HistoricalContext,
)

async def main():
    # Example 1: Simple case - resolved by RULES stage
    print("=" * 60)
    print("Example 1: Low value transaction (should resolve in RULES)")
    print("=" * 60)

    event1 = UniversalEvent(
        id="evt_001",
        domain=DomainType.FINANCE,
        event_type="transaction",
        timestamp=datetime.now(),
        primary_entity="user_123",
        secondary_entity="merchant_456",
        value=5.00,  # Small value
        unit="USD",
        domain_data={}
    )

    context1 = EventContext(
        historical=HistoricalContext(trust_score=0.8)
    )

    result1 = await engine.execute(EventWithContext(event=event1, context=context1))
    print(f"Stages executed: {result1['stages_executed']}")
    print(f"Execution time: {result1['execution_time_ms']:.2f}ms")
    print(f"Stage results: {list(result1['stage_results'].keys())}")
    print()

    # Example 2: Medium case - escalates to ML
    print("=" * 60)
    print("Example 2: Medium value, new account (should escalate to ML)")
    print("=" * 60)

    event2 = UniversalEvent(
        id="evt_002",
        domain=DomainType.FINANCE,
        event_type="transaction",
        timestamp=datetime.now(),
        primary_entity="user_789",
        secondary_entity="merchant_012",
        value=500.00,  # Medium value
        unit="USD",
        domain_data={}
    )

    context2 = EventContext(
        historical=HistoricalContext(
            trust_score=0.6,
            account_age_days=30,
            previous_events_count=10
        )
    )

    result2 = await engine.execute(EventWithContext(event=event2, context=context2))
    print(f"Stages executed: {result2['stages_executed']}")
    print(f"Execution time: {result2['execution_time_ms']:.2f}ms")
    print(f"Stage results: {list(result2['stage_results'].keys())}")
    print()

    # Example 3: Complex case - escalates all the way to AI
    print("=" * 60)
    print("Example 3: High value, risky profile (should escalate to AI)")
    print("=" * 60)

    event3 = UniversalEvent(
        id="evt_003",
        domain=DomainType.FINANCE,
        event_type="transaction",
        timestamp=datetime.now(),
        primary_entity="user_999",
        secondary_entity="merchant_777",
        value=15000.00,  # High value
        unit="USD",
        domain_data={}
    )

    context3 = EventContext(
        historical=HistoricalContext(
            trust_score=0.3,
            account_age_days=7,  # Very new account
            previous_events_count=2
        )
    )

    result3 = await engine.execute(EventWithContext(event=event3, context=context3))
    print(f"Stages executed: {result3['stages_executed']}")
    print(f"Execution time: {result3['execution_time_ms']:.2f}ms")
    print(f"Stage results: {list(result3['stage_results'].keys())}")

    # Print detailed AI reasoning if available
    if "AI" in result3["stage_results"]:
        ai_result = result3["stage_results"]["AI"]
        print(f"\nAI Decision: {ai_result.get('result')}")
        print(f"AI Confidence: {ai_result.get('confidence')}")
        if ai_result.get("data", {}).get("reasoning"):
            print(f"AI Reasoning: {ai_result['data']['reasoning']}")

# Run the cascade
asyncio.run(main())
```

## Expected Output

```
============================================================
Example 1: Low value transaction (should resolve in RULES)
============================================================
Stages executed: 1
Execution time: 0.15ms
Stage results: ['RULES']

============================================================
Example 2: Medium value, new account (should escalate to ML)
============================================================
Stages executed: 2
Execution time: 0.45ms
Stage results: ['RULES', 'ML']

============================================================
Example 3: High value, risky profile (should escalate to AI)
============================================================
Stages executed: 3
Execution time: 1.23ms
Stage results: ['RULES', 'ML', 'AI']

AI Decision: REJECT
AI Confidence: 0.88
AI Reasoning: Multiple risk indicators detected. Activity pattern deviates significantly from expected behavior.
```

## Interpreting the Results

The execution result dictionary contains:

```python
{
    "success": True,                    # Whether execution completed without errors
    "execution_time_ms": 1.23,          # Total execution time
    "stages_executed": 3,               # Number of stages that ran
    "stage_results": {
        "RULES": {
            "stage_name": "RULES",
            "result": "UNCERTAIN",
            "confidence": 0.3,
            "time_ms": 0.12,
            "data": {"reason": "high_value_needs_review"}
        },
        "ML": {
            "stage_name": "ML",
            "result": "UNCERTAIN",
            "confidence": 0.55,
            "time_ms": 0.31,
            "data": {"ml_score": 0.45, "features": {...}}
        },
        "AI": {
            "stage_name": "AI",
            "result": "REJECT",
            "confidence": 0.88,
            "time_ms": 0.80,
            "data": {"reasoning": "...", "factors": [...]}
        }
    },
    "timeline": [...],                   # Execution timeline events
    "routing_decisions": [...]           # Routing decisions made
}
```

### Key Fields

- **stage_results**: Contains the output from each executed stage
- **confidence**: The stage's confidence in its decision (0-1)
- **data**: Stage-specific data (reasoning, features, etc.)
- **routing_decisions**: Shows which stages were enabled and why

## Adding Global Termination Conditions

You can add conditions that terminate the cascade early:

```python
config = CascadeConfig(
    name="my_cascade",
    version="1.0.0",
    stages=stages,
    execution_order=["RULES", "ML", "AI"],
    global_termination_conditions=[
        # Terminate if any stage returns REJECT with high confidence
        Condition(
            operator=ConditionOperator.AND,
            conditions=[
                Condition(
                    field="stages.RULES.result",
                    operator=ConditionOperator.EQ,
                    value="REJECT",
                ),
                Condition(
                    field="stages.RULES.confidence",
                    operator=ConditionOperator.GE,
                    value=0.95,
                ),
            ]
        ),
    ],
    global_timeout_ms=60000,
)
```

## Using Flat Dictionary Input

For backward compatibility, you can also pass a flat dictionary instead of `EventWithContext`:

```python
result = await engine.execute({
    "value": 500.00,
    "trust_score": 0.6,
    "account_age_days": 30,
    "previous_events_count": 10,
    "domain": "FINANCE",
})
```

## Summary

You have now built a complete 3-stage cascade that:

1. Starts with fast, cheap rule-based checks
2. Escalates uncertain cases to ML inference
3. Further escalates edge cases to AI analysis
4. Routes dynamically based on confidence scores
5. Tracks execution timeline and routing decisions

This pattern efficiently handles the 80/20 distribution where most cases can be resolved quickly, while ensuring complex cases get the analysis they need.

## Next Steps

- Learn about [APLS (Adaptive Pattern Learning System)](./apls-learning.md) to optimize costs
- Explore [Event + Context Pattern](../event-context.md) for domain-agnostic design
- Configure [caching and parallel execution](../api/engine.md) for production deployments
