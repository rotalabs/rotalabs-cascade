# Cost Optimization with APLS

The Adaptive Pattern Learning System (APLS) is a cost optimization framework within rotalabs-cascade that learns from expensive stage executions to generate cheaper rules. When your AI stage consistently catches similar patterns, APLS extracts those patterns and proposes rules that can be executed in earlier, cheaper stages.

## The Cost Problem

In a typical cascade system, costs vary dramatically between stages:

| Stage | Relative Cost | Latency | Use Case |
|-------|--------------|---------|----------|
| RULES | 1x | ~1ms | Deterministic checks |
| STATISTICAL_ML | 5x | ~10ms | Trained model inference |
| SINGLE_AI | 25x | ~100ms | Single LLM call |
| POD | 100x | ~500ms | Multi-agent deliberation |
| ADVERSARIAL | 500x | ~2000ms | Adversarial probing |

If 30% of your AI decisions follow predictable patterns, converting those patterns to rules could reduce costs by 95% for that traffic.

## The 5-Level Cost Model

APLS uses a hierarchical cost model defined in `CostAnalyzer`:

```python
from rotalabs_cascade import CostAnalyzer, StageCost

# Default cost configuration
DEFAULT_STAGE_COSTS = {
    "RULES": StageCost(
        stage="RULES",
        base_cost=1.0,      # Baseline cost
        latency_ms=1.0,
        resource_units=0.1,
    ),
    "STATISTICAL_ML": StageCost(
        stage="STATISTICAL_ML",
        base_cost=5.0,      # 5x rules
        latency_ms=10.0,
        resource_units=0.5,
    ),
    "SINGLE_AI": StageCost(
        stage="SINGLE_AI",
        base_cost=25.0,     # 25x rules
        latency_ms=100.0,
        resource_units=2.0,
    ),
    "POD": StageCost(
        stage="POD",
        base_cost=100.0,    # 100x rules
        latency_ms=500.0,
        resource_units=10.0,
    ),
    "ADVERSARIAL": StageCost(
        stage="ADVERSARIAL",
        base_cost=500.0,    # 500x rules
        latency_ms=2000.0,
        resource_units=50.0,
    ),
}
```

### Customizing Stage Costs

You can configure costs to match your actual infrastructure:

```python
from rotalabs_cascade import CostAnalyzer, StageCost

# Create analyzer with default costs
analyzer = CostAnalyzer()

# Add or update a custom stage
analyzer.set_stage_cost("CUSTOM_STAGE", StageCost(
    stage="CUSTOM_STAGE",
    base_cost=15.0,
    latency_ms=50.0,
    resource_units=1.5,
))

# Or initialize with all custom costs
custom_costs = {
    "RULES": StageCost(stage="RULES", base_cost=1.0, latency_ms=1.0, resource_units=0.1),
    "ML": StageCost(stage="ML", base_cost=10.0, latency_ms=20.0, resource_units=1.0),
    "AI": StageCost(stage="AI", base_cost=50.0, latency_ms=200.0, resource_units=5.0),
}
analyzer = CostAnalyzer(stage_costs=custom_costs)
```

## APLS Components

The APLS system consists of four main components:

1. **PatternExtractor**: Learns patterns from stage failures/detections
2. **RuleGenerator**: Converts patterns into routing rules
3. **CostAnalyzer**: Calculates ROI for pattern migrations
4. **ProposalManager**: Manages the human approval workflow

## Component 1: PatternExtractor

The `PatternExtractor` observes stage executions and extracts recurring patterns that could be moved to earlier stages.

### Pattern Types

APLS extracts five types of patterns:

| Type | Description | Example |
|------|-------------|---------|
| `threshold` | Simple value comparisons | `amount > 5000` |
| `correlation` | Feature combinations | `new_account AND high_value` |
| `reasoning` | AI explanation patterns | Keywords and factors from AI reasoning |
| `temporal` | Time-based patterns | Peak hours, processing time thresholds |
| `behavioral` | Execution flow patterns | Stage sequences, routing decisions |

### Using PatternExtractor

```python
from rotalabs_cascade import (
    PatternExtractor,
    PatternConfig,
    ExecutionContext,
    StageResult,
)

# Configure the extractor
config = PatternConfig(
    min_confidence=0.7,           # Minimum confidence to extract pattern
    min_samples_for_candidate=10, # Samples needed before migration consideration
    max_patterns=1000,            # Maximum patterns to track (LRU eviction)
    pattern_ttl_hours=168,        # Patterns expire after 7 days without updates
    enable_threshold_extraction=True,
    enable_correlation_extraction=True,
    enable_reasoning_extraction=True,
    enable_temporal_extraction=True,
    enable_behavioral_extraction=True,
)

# Create the extractor
extractor = PatternExtractor(config)
```

### Learning from Stage Executions

After each cascade execution, feed the results to the extractor:

```python
async def process_and_learn(engine, extractor, event_with_context):
    """Execute cascade and learn from results."""
    # Execute the cascade
    result = await engine.execute(event_with_context)

    # Create execution context for learning
    context = ExecutionContext(event_with_context)

    # Learn from each stage that executed
    for stage_name, stage_data in result["stage_results"].items():
        stage_result = StageResult(
            stage_name=stage_name,
            result=stage_data.get("result"),
            confidence=stage_data.get("confidence"),
            data=stage_data.get("data", {}),
            time_ms=stage_data.get("time_ms", 0),
        )

        # Learn from stages that made decisions (not just passed through)
        if stage_result.result in ("REJECT", "FLAG", "ESCALATE"):
            pattern = extractor.learn_from_failure(
                context=context,
                stage=stage_name,
                result=stage_result,
            )
            if pattern:
                print(f"Learned pattern {pattern.id} from {stage_name}")

    return result
```

### Getting Migration Candidates

Once enough patterns accumulate, query for migration candidates:

```python
# Get patterns ready for migration
candidates = extractor.get_migration_candidates(
    min_confidence=0.8,
    min_samples=10,
)

print(f"Found {len(candidates)} migration candidates")

for pattern in candidates:
    print(f"  Pattern {pattern.id}:")
    print(f"    Stage: {pattern.stage}")
    print(f"    Type: {pattern.pattern_type}")
    print(f"    Confidence: {pattern.confidence:.2%}")
    print(f"    Samples: {pattern.sample_count}")
```

### Getting Insights

Get aggregated insights about learned patterns:

```python
insights = extractor.get_insights()

print(f"Total patterns: {len(insights.patterns)}")
print(f"Migration candidates: {len(insights.migration_candidates)}")
print(f"Estimated cost reduction: {insights.estimated_cost_reduction:.1%}")
print(f"Detection rate impact: {insights.detection_rate_impact:.2%}")
```

## Component 2: RuleGenerator

The `RuleGenerator` converts extracted patterns into routing rules that can be deployed to earlier stages.

### Generating Rules from Patterns

```python
from rotalabs_cascade import RuleGenerator

# Create the generator
generator = RuleGenerator(
    min_confidence=0.7,    # Minimum pattern confidence to generate rule
    min_coverage=0.01,     # Minimum estimated coverage
)

# Generate rules from all candidates
rules = []
for pattern in candidates:
    rule = generator.generate_from_pattern(pattern)
    if rule:
        rules.append(rule)
        print(f"Generated rule: {rule.name}")
        print(f"  Template: {rule.template.value}")
        print(f"  Conditions: {rule.conditions}")
        print(f"  Action: {rule.action}")
```

### Rule Templates

The generator produces different rule templates based on pattern type:

```python
from rotalabs_cascade import RuleTemplate

# VALUE_THRESHOLD - Simple numeric comparisons
# Generated from: threshold patterns
# Example: value >= 5000

# MULTI_CONDITION - AND/OR combinations
# Generated from: correlation and reasoning patterns
# Example: new_account AND high_value AND unusual_location

# PATTERN_MATCH - Regex matching
# Generated from: reasoning patterns with keywords
# Example: content MATCHES "buy|sell|offer"

# TEMPORAL - Time-based rules
# Generated from: temporal patterns
# Example: hour >= 22 OR hour <= 6

# BEHAVIORAL - Execution flow rules
# Generated from: behavioral patterns
# Example: stage_count >= 3 AND avg_confidence < 0.6
```

### Converting to RoutingRule

Generated rules can be converted to the cascade's native `RoutingRule` format:

```python
# Convert GeneratedRule to RoutingRule for cascade config
for generated_rule in rules:
    routing_rule = generator.to_routing_rule(generated_rule)
    print(f"Routing rule: {routing_rule.name}")
    print(f"  Type: {routing_rule.type}")
    print(f"  Priority: {routing_rule.priority}")
```

### Batch Processing

Generate rules from multiple patterns at once:

```python
# Generate rules from all patterns
rules = generator.generate_batch(candidates)
print(f"Generated {len(rules)} rules from {len(candidates)} patterns")
```

### Exporting to YAML

Export generated rules for review:

```python
yaml_output = generator.to_yaml(rules)
print(yaml_output)
```

Output:

```yaml
generated_rules:
  - rule_id: rule_abc123
    name: threshold_AI_value
    description: Route when value is greater than or equal to 5000...
    template: value_threshold
    conditions:
      - field: value
        operator: ">="
        value: 5000
    action: APPROVE
    target_stage: RULES
    source_pattern_id: pat_xyz
    confidence: 0.92
    estimated_coverage: 0.08
    created_at: '2024-01-15T10:30:00+00:00'
metadata:
  generated_at: '2024-01-15T10:30:00+00:00'
  rule_count: 5
  templates_used:
    - value_threshold
    - multi_condition
```

## Component 3: CostAnalyzer

The `CostAnalyzer` calculates the ROI of migrating patterns to cheaper stages.

### Calculating Migration ROI

```python
from rotalabs_cascade import CostAnalyzer, MigrationROI

analyzer = CostAnalyzer()

# Calculate ROI for a single pattern
for pattern in candidates:
    roi = analyzer.calculate_migration_roi(
        pattern=pattern,
        target_stage="RULES",      # Migrate to RULES stage
        volume=10000,              # Expected events per period
        migration_effort=100.0,    # One-time migration cost
    )

    print(f"Pattern {pattern.id} -> RULES:")
    print(f"  Current cost/item: {roi.current_cost_per_item:.2f}")
    print(f"  Projected cost/item: {roi.projected_cost_per_item:.2f}")
    print(f"  Cost reduction: {roi.cost_reduction_percentage:.1f}%")
    print(f"  Absolute savings: {roi.cost_reduction_absolute:.2f}")
    print(f"  False positive risk: {roi.false_positive_risk:.2%}")
    print(f"  Payback items: {roi.payback_items}")
    print(f"  Recommendation: {roi.recommendation}")
```

### MigrationROI Fields

The `MigrationROI` dataclass contains:

```python
@dataclass
class MigrationROI:
    pattern_id: str                  # Pattern being analyzed
    source_stage: str                # Current stage
    target_stage: str                # Proposed target stage
    current_cost_per_item: float     # Cost at current stage
    projected_cost_per_item: float   # Cost at target stage
    estimated_volume: int            # Expected items per period
    cost_reduction_absolute: float   # Total savings per period
    cost_reduction_percentage: float # Percentage reduction
    detection_rate_change: float     # Impact on detection rate
    false_positive_risk: float       # Risk of false positives (0-1)
    payback_items: int               # Items to break even
    recommendation: str              # MIGRATE, MONITOR, or REJECT
```

### Recommendations

The analyzer provides three recommendations:

- **MIGRATE**: High confidence pattern with significant savings (>20% reduction, >80% confidence)
- **MONITOR**: Moderate savings or confidence, worth tracking
- **REJECT**: Too risky (high false positive risk) or insufficient savings (<5%)

### Analyzing All Candidates

```python
# Analyze all candidates for migration to RULES
roi_results = analyzer.analyze_all_candidates(
    candidates=candidates,
    target_stage="RULES",
    volume=10000,
)

# Get total potential savings
savings = analyzer.get_total_potential_savings(roi_results)

print(f"Total potential savings: {savings['total_absolute_savings']:.2f}")
print(f"Average reduction: {savings['average_reduction_percentage']:.1f}%")
print(f"Patterns to migrate: {savings['migrate_count']}")
print(f"Patterns to monitor: {savings['monitor_count']}")
print(f"Patterns rejected: {savings['reject_count']}")
```

### Ranking by ROI

Get the highest-impact migrations first:

```python
# Rank by ROI (highest savings first)
ranked = analyzer.rank_by_roi(roi_results, ascending=False)

print("Top 5 migration opportunities:")
for roi in ranked[:5]:
    print(f"  {roi.pattern_id}: {roi.cost_reduction_absolute:.2f} savings")
```

## Component 4: ProposalManager

The `ProposalManager` handles the human-in-the-loop approval workflow for generated rules.

### Proposal Lifecycle

```
PENDING_REVIEW -> APPROVED -> TESTING -> ACTIVE
                     |
                     v
                  REJECTED

ACTIVE -> DEPRECATED
```

### Creating Proposals

```python
from rotalabs_cascade import ProposalManager
from pathlib import Path

# Create manager with optional persistence
manager = ProposalManager(storage_path=Path("./proposals"))

# Create proposals from generated rules and ROI analysis
for rule, roi in zip(rules, roi_results):
    if roi.recommendation == "MIGRATE":
        proposal = manager.create_proposal(
            rule=rule,
            roi=roi,
        )
        print(f"Created proposal {proposal.proposal_id}")
```

### Reviewing Proposals

```python
# Get pending proposals
pending = manager.get_pending_proposals()

for proposal in pending:
    print(f"Proposal {proposal.proposal_id}:")
    print(f"  Rule: {proposal.generated_rule.name}")
    print(f"  Confidence: {proposal.generated_rule.confidence:.2%}")
    print(f"  Savings: {proposal.roi_analysis.cost_reduction_absolute:.2f}")
    print(f"  Status: {proposal.status.value}")
```

### Approving or Rejecting

```python
# Approve a proposal
manager.approve(
    proposal_id=proposal.proposal_id,
    reviewer="analyst@example.com",
    notes="Pattern verified against historical data. Safe to proceed."
)

# Reject a proposal
manager.reject(
    proposal_id=other_proposal.proposal_id,
    reviewer="analyst@example.com",
    notes="Pattern too broad, high false positive risk in edge cases."
)
```

### A/B Testing

```python
# Start A/B testing
manager.start_testing(proposal.proposal_id)

# Record test results over time
manager.record_test_results(
    proposal_id=proposal.proposal_id,
    results={
        "accuracy": 0.94,
        "false_positive_rate": 0.02,
        "samples_tested": 1500,
        "test_duration_hours": 48,
    }
)
```

### Activation and Deprecation

```python
# Activate approved and tested rule
manager.activate(proposal.proposal_id)

# Get all active rules
active_rules = manager.get_active_rules()

# Later: deprecate if no longer needed
manager.deprecate(
    proposal_id=proposal.proposal_id,
    reason="Pattern no longer relevant after model update."
)
```

### Export and Import

```python
# Export all proposals
manager.export_proposals(Path("./proposals_backup.json"))

# Import proposals
manager.import_proposals(Path("./proposals_backup.json"))
```

## Complete APLS Workflow

Here's a complete example bringing all components together:

```python
import asyncio
from pathlib import Path
from datetime import datetime
from rotalabs_cascade import (
    CascadeEngine,
    CascadeConfig,
    StageConfig,
    ExecutionContext,
    StageResult,
    PatternExtractor,
    PatternConfig,
    RuleGenerator,
    CostAnalyzer,
    ProposalManager,
    EventWithContext,
    UniversalEvent,
    EventContext,
    DomainType,
    HistoricalContext,
)

async def run_apls_workflow():
    # 1. Set up the cascade engine (from previous tutorial)
    engine = create_cascade_engine()

    # 2. Set up APLS components
    extractor = PatternExtractor(PatternConfig(
        min_confidence=0.7,
        min_samples_for_candidate=10,
    ))

    generator = RuleGenerator(min_confidence=0.7)

    analyzer = CostAnalyzer()

    manager = ProposalManager(storage_path=Path("./proposals"))

    # 3. Process events and learn patterns
    print("Processing events and learning patterns...")

    for i in range(100):  # Simulate 100 events
        event = generate_random_event(i)
        result = await engine.execute(event)

        # Learn from AI stage executions
        if "AI" in result["stage_results"]:
            context = ExecutionContext(event)
            ai_data = result["stage_results"]["AI"]
            stage_result = StageResult(
                stage_name="AI",
                result=ai_data.get("result"),
                confidence=ai_data.get("confidence"),
                data=ai_data.get("data", {}),
                time_ms=ai_data.get("time_ms", 0),
            )
            extractor.learn_from_failure(context, "AI", stage_result)

    # 4. Get migration candidates
    print("\nFinding migration candidates...")
    candidates = extractor.get_migration_candidates(
        min_confidence=0.8,
        min_samples=5,
    )
    print(f"Found {len(candidates)} candidates")

    # 5. Generate rules
    print("\nGenerating rules...")
    rules = generator.generate_batch(candidates)
    print(f"Generated {len(rules)} rules")

    # 6. Analyze ROI
    print("\nAnalyzing ROI...")
    roi_results = analyzer.analyze_all_candidates(
        candidates=candidates,
        target_stage="RULES",
        volume=10000,
    )

    savings = analyzer.get_total_potential_savings(roi_results)
    print(f"Total potential savings: {savings['total_absolute_savings']:.2f}")
    print(f"Patterns to migrate: {savings['migrate_count']}")

    # 7. Create proposals for MIGRATE recommendations
    print("\nCreating proposals...")
    for rule, roi in zip(rules, roi_results):
        if roi.recommendation == "MIGRATE":
            proposal = manager.create_proposal(rule, roi)
            print(f"  Created: {proposal.proposal_id} - {rule.name}")

    # 8. Simulate human review
    print("\nPending proposals for review:")
    pending = manager.get_pending_proposals()
    for p in pending:
        print(f"  {p.proposal_id}: {p.generated_rule.name}")

    print("\nAPLS workflow complete!")
    return manager


def generate_random_event(seed: int) -> EventWithContext:
    """Generate a random event for testing."""
    import random
    random.seed(seed)

    event = UniversalEvent(
        id=f"evt_{seed:04d}",
        domain=DomainType.FINANCE,
        event_type="transaction",
        timestamp=datetime.now(),
        primary_entity=f"user_{random.randint(1, 100)}",
        secondary_entity=f"merchant_{random.randint(1, 50)}",
        value=random.uniform(10, 10000),
        unit="USD",
        domain_data={}
    )

    context = EventContext(
        historical=HistoricalContext(
            account_age_days=random.randint(1, 1000),
            previous_events_count=random.randint(0, 500),
            trust_score=random.uniform(0.2, 0.95),
        )
    )

    return EventWithContext(event=event, context=context)


# Run the workflow
asyncio.run(run_apls_workflow())
```

## Best Practices

### 1. Start with High-Confidence Patterns

Only migrate patterns with confidence >= 0.8 to minimize false positives:

```python
candidates = extractor.get_migration_candidates(
    min_confidence=0.85,  # Start conservative
    min_samples=20,       # Require more evidence
)
```

### 2. Use A/B Testing

Always test before full activation:

```python
# Test with 10% traffic first
manager.start_testing(proposal_id)
# ... run tests ...
manager.record_test_results(proposal_id, {
    "traffic_percentage": 10,
    "accuracy": 0.96,
    "false_positive_rate": 0.01,
})
```

### 3. Monitor Detection Rate Impact

Watch for patterns that reduce detection effectiveness:

```python
roi = analyzer.calculate_migration_roi(pattern, "RULES", volume=10000)
if roi.detection_rate_change < -0.05:  # >5% detection rate drop
    print(f"Warning: Migration may reduce detection by {-roi.detection_rate_change:.1%}")
```

### 4. Set Up Automatic Pattern Expiration

Patterns should expire if not seen recently:

```python
config = PatternConfig(
    pattern_ttl_hours=168,  # 7 days
)
```

### 5. Export Rules for Code Review

Generate YAML for human review:

```python
yaml_output = generator.to_yaml(rules)
with open("proposed_rules.yaml", "w") as f:
    f.write(yaml_output)
```

## Summary

APLS provides a complete framework for learning from expensive AI decisions and migrating patterns to cheaper rule-based stages:

1. **PatternExtractor** learns from stage executions
2. **RuleGenerator** converts patterns to rules
3. **CostAnalyzer** calculates migration ROI
4. **ProposalManager** handles human approval

The result is continuous cost optimization while maintaining detection quality through human oversight and A/B testing.
