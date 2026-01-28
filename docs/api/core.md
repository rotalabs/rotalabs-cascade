# Core Module

The core module provides the foundational classes for cascade orchestration, including the execution engine, configuration schema, and execution context management.

## CascadeEngine

::: rotalabs_cascade.core.engine.CascadeEngine
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - register_stage
        - execute
        - get_statistics
        - clear_cache
        - update_config

## CascadeConfig

::: rotalabs_cascade.core.config.CascadeConfig
    options:
      show_source: false
      heading_level: 3
      members:
        - from_dict
        - from_file
        - to_dict
        - to_json
        - to_yaml

## StageConfig

::: rotalabs_cascade.core.config.StageConfig
    options:
      show_source: false
      heading_level: 3
      members:
        - from_dict
        - to_dict

## RoutingRule

::: rotalabs_cascade.core.config.RoutingRule
    options:
      show_source: false
      heading_level: 3
      members:
        - from_dict
        - to_dict

## Condition

::: rotalabs_cascade.core.config.Condition
    options:
      show_source: false
      heading_level: 3
      members:
        - from_dict
        - to_dict

## ConditionOperator

::: rotalabs_cascade.core.config.ConditionOperator
    options:
      show_source: false
      heading_level: 3

## ExecutionContext

::: rotalabs_cascade.core.context.ExecutionContext
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - data
        - elapsed_ms
        - should_terminate
        - get
        - set
        - add_stage_result
        - add_stage_error
        - get_stage_result
        - set_termination_flag
        - set_next_stage
        - get_next_stage
        - enable_stage
        - disable_stage
        - is_stage_enabled
        - set_metadata
        - get_metadata
        - get_result

## StageResult

::: rotalabs_cascade.core.context.StageResult
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - to_dict
