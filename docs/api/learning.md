# Learning Module (APLS)

The learning module implements the Automatic Pattern Learning System (APLS) for cascade optimization. It extracts patterns from stage executions, generates routing rules, analyzes migration costs, and manages the rule proposal workflow.

## PatternExtractor

::: rotalabs_cascade.learning.pattern_extractor.PatternExtractor
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - learn_from_failure
        - get_migration_candidates
        - get_insights
        - clear_patterns
        - get_pattern
        - get_patterns_by_stage
        - get_patterns_by_type
        - export_patterns
        - import_patterns

## StageFailurePattern

::: rotalabs_cascade.learning.pattern_extractor.StageFailurePattern
    options:
      show_source: false
      heading_level: 3
      members:
        - to_dict
        - from_dict

## RuleGenerator

::: rotalabs_cascade.learning.rule_generator.RuleGenerator
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - generate_from_pattern
        - to_routing_rule
        - to_yaml
        - generate_batch

## GeneratedRule

::: rotalabs_cascade.learning.rule_generator.GeneratedRule
    options:
      show_source: false
      heading_level: 3
      members:
        - to_dict
        - from_dict

## CostAnalyzer

::: rotalabs_cascade.learning.cost_analyzer.CostAnalyzer
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - set_stage_cost
        - calculate_migration_roi
        - analyze_all_candidates
        - get_total_potential_savings
        - rank_by_roi

## MigrationROI

::: rotalabs_cascade.learning.cost_analyzer.MigrationROI
    options:
      show_source: false
      heading_level: 3

## ProposalManager

::: rotalabs_cascade.learning.proposal.ProposalManager
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - create_proposal
        - get_pending_proposals
        - approve
        - reject
        - start_testing
        - record_test_results
        - activate
        - deprecate
        - get_proposal
        - get_active_rules
        - export_proposals
        - import_proposals

## RuleProposal

::: rotalabs_cascade.learning.proposal.RuleProposal
    options:
      show_source: false
      heading_level: 3
      members:
        - to_dict
        - from_dict
