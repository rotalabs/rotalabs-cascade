"""Tests for configuration classes in rotalabs-cascade.

Tests cover:
- Condition creation and serialization
- RoutingAction validation
- RoutingRule structure
- StageConfig with defaults and custom values
- CascadeConfig serialization (JSON/YAML)
"""

import json
import pytest
from typing import Any, Dict

from rotalabs_cascade.core.config import (
    CascadeConfig,
    Condition,
    ConditionOperator,
    RoutingAction,
    RoutingRule,
    StageConfig,
)


class TestCondition:
    """Tests for Condition class."""

    def test_simple_condition_creation(self):
        """Test creating a simple field-based condition."""
        condition = Condition(
            field="confidence",
            operator=ConditionOperator.GT,
            value=0.8,
        )

        assert condition.field == "confidence"
        assert condition.operator == ConditionOperator.GT
        assert condition.value == 0.8
        assert condition.conditions is None

    def test_condition_with_string_operator(self):
        """Test that string operators are converted to ConditionOperator enum."""
        condition = Condition(
            field="status",
            operator="==",
            value="ready",
        )

        assert condition.operator == ConditionOperator.EQ

    def test_composite_condition_validation(self):
        """Test that composite operators require nested conditions."""
        with pytest.raises(ValueError, match="AND requires nested conditions"):
            Condition(
                operator=ConditionOperator.AND,
                # Missing conditions
            )

    def test_field_operator_validation(self):
        """Test that field-based operators require a field."""
        with pytest.raises(ValueError, match="requires a field"):
            Condition(
                operator=ConditionOperator.EQ,
                value=5,
                # Missing field
            )

    def test_condition_to_dict(self):
        """Test converting condition to dictionary."""
        condition = Condition(
            field="score",
            operator=ConditionOperator.GE,
            value=100,
        )

        result = condition.to_dict()

        assert result == {
            "field": "score",
            "operator": ">=",
            "value": 100,
        }

    def test_condition_from_dict(self):
        """Test creating condition from dictionary."""
        data = {
            "field": "status",
            "operator": "IN",
            "value": ["active", "pending"],
        }

        condition = Condition.from_dict(data)

        assert condition.field == "status"
        assert condition.operator == ConditionOperator.IN
        assert condition.value == ["active", "pending"]

    def test_composite_condition_serialization(self):
        """Test serializing and deserializing composite conditions."""
        condition = Condition(
            operator=ConditionOperator.AND,
            conditions=[
                Condition(field="x", operator=ConditionOperator.GT, value=5),
                Condition(field="y", operator=ConditionOperator.LT, value=10),
            ],
        )

        data = condition.to_dict()
        restored = Condition.from_dict(data)

        assert restored.operator == ConditionOperator.AND
        assert len(restored.conditions) == 2
        assert restored.conditions[0].field == "x"
        assert restored.conditions[1].field == "y"

    def test_nested_composite_conditions(self):
        """Test deeply nested composite conditions."""
        condition = Condition(
            operator=ConditionOperator.OR,
            conditions=[
                Condition(field="a", operator=ConditionOperator.EQ, value=1),
                Condition(
                    operator=ConditionOperator.AND,
                    conditions=[
                        Condition(field="b", operator=ConditionOperator.GT, value=2),
                        Condition(field="c", operator=ConditionOperator.LT, value=5),
                    ],
                ),
            ],
        )

        data = condition.to_dict()
        restored = Condition.from_dict(data)

        assert restored.operator == ConditionOperator.OR
        assert len(restored.conditions) == 2
        assert restored.conditions[1].operator == ConditionOperator.AND


class TestRoutingAction:
    """Tests for RoutingAction class."""

    def test_terminate_action(self):
        """Test creating a terminate action."""
        action = RoutingAction(type="terminate")

        assert action.type == "terminate"
        assert action.target is None
        assert action.stages is None

    def test_skip_to_action(self):
        """Test creating a skip_to action."""
        action = RoutingAction(type="skip_to", target="FINAL_STAGE")

        assert action.type == "skip_to"
        assert action.target == "FINAL_STAGE"

    def test_skip_to_requires_target(self):
        """Test that skip_to action requires target."""
        with pytest.raises(ValueError, match="skip_to action requires target"):
            RoutingAction(type="skip_to")

    def test_enable_stages_action(self):
        """Test creating an enable_stages action."""
        action = RoutingAction(
            type="enable_stages",
            stages=["STAGE_A", "STAGE_B"],
        )

        assert action.type == "enable_stages"
        assert action.stages == ["STAGE_A", "STAGE_B"]

    def test_enable_stages_requires_stages(self):
        """Test that enable_stages requires stages list."""
        with pytest.raises(ValueError, match="enable_stages action requires stages"):
            RoutingAction(type="enable_stages")

    def test_disable_stages_action(self):
        """Test creating a disable_stages action."""
        action = RoutingAction(
            type="disable_stages",
            stages=["STAGE_C"],
        )

        assert action.type == "disable_stages"
        assert action.stages == ["STAGE_C"]

    def test_set_field_action(self):
        """Test creating a set_field action."""
        action = RoutingAction(
            type="set_field",
            field="result.status",
            value="completed",
        )

        assert action.type == "set_field"
        assert action.field == "result.status"
        assert action.value == "completed"

    def test_set_field_requires_field(self):
        """Test that set_field requires field."""
        with pytest.raises(ValueError, match="set_field action requires field"):
            RoutingAction(type="set_field", value="test")

    def test_invalid_action_type(self):
        """Test that invalid action types are rejected."""
        with pytest.raises(ValueError, match="Invalid action type"):
            RoutingAction(type="invalid_type")

    def test_action_to_dict(self):
        """Test converting action to dictionary."""
        action = RoutingAction(
            type="enable_stages",
            stages=["STAGE_X", "STAGE_Y"],
        )

        result = action.to_dict()

        assert result == {
            "type": "enable_stages",
            "stages": ["STAGE_X", "STAGE_Y"],
        }

    def test_action_from_dict(self):
        """Test creating action from dictionary."""
        data = {
            "type": "skip_to",
            "target": "FINAL",
        }

        action = RoutingAction.from_dict(data)

        assert action.type == "skip_to"
        assert action.target == "FINAL"


class TestRoutingRule:
    """Tests for RoutingRule class."""

    def test_routing_rule_creation(self):
        """Test creating a routing rule."""
        condition = Condition(
            field="confidence",
            operator=ConditionOperator.LT,
            value=0.5,
        )
        action = RoutingAction(type="terminate")

        rule = RoutingRule(
            name="low_confidence_abort",
            type="precondition",
            condition=condition,
            action=action,
            priority=10,
        )

        assert rule.name == "low_confidence_abort"
        assert rule.type == "precondition"
        assert rule.priority == 10

    def test_rule_default_priority(self):
        """Test that routing rule has default priority of 0."""
        rule = RoutingRule(
            name="test_rule",
            type="routing",
            condition=Condition(field="x", operator=ConditionOperator.GT, value=0),
            action=RoutingAction(type="terminate"),
        )

        assert rule.priority == 0

    def test_invalid_rule_type(self):
        """Test that invalid rule types are rejected."""
        with pytest.raises(ValueError, match="Invalid rule type"):
            RoutingRule(
                name="test",
                type="invalid_type",
                condition=Condition(field="x", operator=ConditionOperator.GT, value=0),
                action=RoutingAction(type="terminate"),
            )

    def test_rule_serialization(self):
        """Test routing rule to_dict and from_dict."""
        rule = RoutingRule(
            name="test_rule",
            type="postcondition",
            condition=Condition(field="status", operator=ConditionOperator.EQ, value="done"),
            action=RoutingAction(type="terminate"),
            priority=5,
        )

        data = rule.to_dict()
        restored = RoutingRule.from_dict(data)

        assert restored.name == "test_rule"
        assert restored.type == "postcondition"
        assert restored.priority == 5
        assert restored.condition.field == "status"
        assert restored.action.type == "terminate"


class TestStageConfig:
    """Tests for StageConfig class."""

    def test_minimal_stage_config(self):
        """Test creating a stage with minimal configuration."""
        stage = StageConfig(name="TEST_STAGE")

        assert stage.name == "TEST_STAGE"
        assert stage.enabled is True
        assert stage.timeout_ms == 30000
        assert stage.max_retries == 0
        assert stage.depends_on == []
        assert stage.routing_rules == []

    def test_stage_with_custom_values(self):
        """Test creating a stage with custom configuration."""
        stage = StageConfig(
            name="CUSTOM_STAGE",
            enabled=False,
            timeout_ms=5000,
            max_retries=3,
            retry_delay_ms=500,
            can_run_parallel=True,
            parallel_group="group_a",
        )

        assert stage.name == "CUSTOM_STAGE"
        assert stage.enabled is False
        assert stage.timeout_ms == 5000
        assert stage.max_retries == 3
        assert stage.retry_delay_ms == 500
        assert stage.can_run_parallel is True
        assert stage.parallel_group == "group_a"

    def test_stage_with_dependencies(self):
        """Test stage with dependencies."""
        stage = StageConfig(
            name="DEPENDENT_STAGE",
            depends_on=["STAGE_A", "STAGE_B"],
        )

        assert stage.depends_on == ["STAGE_A", "STAGE_B"]

    def test_stage_with_routing_rules(self):
        """Test stage with routing rules."""
        rule = RoutingRule(
            name="test_rule",
            type="routing",
            condition=Condition(field="x", operator=ConditionOperator.GT, value=0),
            action=RoutingAction(type="terminate"),
        )

        stage = StageConfig(
            name="STAGE_WITH_RULES",
            routing_rules=[rule],
        )

        assert len(stage.routing_rules) == 1
        assert stage.routing_rules[0].name == "test_rule"

    def test_stage_caching_config(self):
        """Test stage caching configuration."""
        stage = StageConfig(
            name="CACHED_STAGE",
            cache_enabled=True,
            cache_ttl_seconds=600,
        )

        assert stage.cache_enabled is True
        assert stage.cache_ttl_seconds == 600

    def test_stage_custom_properties(self):
        """Test stage with custom properties."""
        stage = StageConfig(
            name="CUSTOM_PROPS_STAGE",
            custom_properties={
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        )

        assert stage.custom_properties["model"] == "gpt-4"
        assert stage.custom_properties["temperature"] == 0.7

    def test_stage_serialization(self):
        """Test stage config to_dict and from_dict."""
        stage = StageConfig(
            name="TEST_STAGE",
            enabled=False,
            timeout_ms=10000,
            max_retries=2,
            depends_on=["STAGE_A"],
        )

        data = stage.to_dict()
        restored = StageConfig.from_dict(data)

        assert restored.name == "TEST_STAGE"
        assert restored.enabled is False
        assert restored.timeout_ms == 10000
        assert restored.max_retries == 2
        assert restored.depends_on == ["STAGE_A"]


class TestCascadeConfig:
    """Tests for CascadeConfig class."""

    def test_minimal_cascade_config(self):
        """Test creating a minimal cascade configuration."""
        config = CascadeConfig(
            name="test_cascade",
            version="1.0.0",
            stages={
                "STAGE_A": StageConfig(name="STAGE_A"),
            },
        )

        assert config.name == "test_cascade"
        assert config.version == "1.0.0"
        assert "STAGE_A" in config.stages
        assert config.global_timeout_ms == 300000
        assert config.max_parallel_stages == 5

    def test_execution_order_validation(self):
        """Test that execution order references existing stages."""
        with pytest.raises(ValueError, match="references unknown stage"):
            CascadeConfig(
                name="test",
                version="1.0.0",
                stages={"STAGE_A": StageConfig(name="STAGE_A")},
                execution_order=["STAGE_A", "NONEXISTENT"],
            )

    def test_dependency_validation(self):
        """Test that stage dependencies reference existing stages."""
        with pytest.raises(ValueError, match="depends on unknown stage"):
            CascadeConfig(
                name="test",
                version="1.0.0",
                stages={
                    "STAGE_A": StageConfig(
                        name="STAGE_A",
                        depends_on=["NONEXISTENT"],
                    ),
                },
            )

    def test_cascade_with_execution_order(self):
        """Test cascade with explicit execution order."""
        config = CascadeConfig(
            name="ordered_cascade",
            version="1.0.0",
            stages={
                "FIRST": StageConfig(name="FIRST"),
                "SECOND": StageConfig(name="SECOND"),
                "THIRD": StageConfig(name="THIRD"),
            },
            execution_order=["FIRST", "SECOND", "THIRD"],
        )

        assert config.execution_order == ["FIRST", "SECOND", "THIRD"]

    def test_cascade_with_termination_conditions(self):
        """Test cascade with global termination conditions."""
        config = CascadeConfig(
            name="cascade_with_term",
            version="1.0.0",
            stages={"STAGE": StageConfig(name="STAGE")},
            global_termination_conditions=[
                Condition(
                    field="error_count",
                    operator=ConditionOperator.GT,
                    value=5,
                ),
            ],
        )

        assert len(config.global_termination_conditions) == 1

    def test_cascade_caching_config(self):
        """Test cascade caching configuration."""
        config = CascadeConfig(
            name="cached_cascade",
            version="1.0.0",
            stages={"STAGE": StageConfig(name="STAGE")},
            enable_caching=True,
            cache_key_fields=["user_id", "request_id"],
        )

        assert config.enable_caching is True
        assert config.cache_key_fields == ["user_id", "request_id"]

    def test_cascade_domain_config(self):
        """Test cascade domain-specific configuration."""
        config = CascadeConfig(
            name="domain_cascade",
            version="1.0.0",
            stages={"STAGE": StageConfig(name="STAGE")},
            domain_config={
                "llm_provider": "openai",
                "default_model": "gpt-4",
            },
        )

        assert config.domain_config["llm_provider"] == "openai"

    def test_cascade_to_dict(self):
        """Test converting cascade config to dictionary."""
        config = CascadeConfig(
            name="test_cascade",
            version="1.0.0",
            stages={
                "STAGE_A": StageConfig(name="STAGE_A"),
                "STAGE_B": StageConfig(name="STAGE_B"),
            },
            execution_order=["STAGE_A", "STAGE_B"],
        )

        data = config.to_dict()

        assert data["name"] == "test_cascade"
        assert data["version"] == "1.0.0"
        assert "STAGE_A" in data["stages"]
        assert data["execution_order"] == ["STAGE_A", "STAGE_B"]

    def test_cascade_from_dict(self):
        """Test creating cascade config from dictionary."""
        data = {
            "name": "test_cascade",
            "version": "2.0.0",
            "stages": {
                "STAGE_A": {
                    "name": "STAGE_A",
                    "enabled": True,
                    "handler_type": None,
                    "timeout_ms": 30000,
                    "max_retries": 0,
                    "retry_delay_ms": 1000,
                    "can_run_parallel": False,
                    "parallel_group": None,
                    "depends_on": [],
                    "routing_rules": [],
                    "cache_enabled": False,
                    "cache_ttl_seconds": 3600,
                    "custom_properties": {},
                },
            },
            "execution_order": ["STAGE_A"],
            "global_timeout_ms": 60000,
            "max_parallel_stages": 3,
            "global_termination_conditions": [],
            "enable_caching": False,
            "cache_key_fields": [],
            "domain_config": {},
        }

        config = CascadeConfig.from_dict(data)

        assert config.name == "test_cascade"
        assert config.version == "2.0.0"
        assert config.global_timeout_ms == 60000
        assert config.max_parallel_stages == 3

    def test_cascade_to_json(self):
        """Test converting cascade config to JSON."""
        config = CascadeConfig(
            name="json_cascade",
            version="1.0.0",
            stages={"STAGE": StageConfig(name="STAGE")},
        )

        json_str = config.to_json()
        parsed = json.loads(json_str)

        assert parsed["name"] == "json_cascade"
        assert parsed["version"] == "1.0.0"

    def test_cascade_to_yaml(self):
        """Test converting cascade config to YAML."""
        config = CascadeConfig(
            name="yaml_cascade",
            version="1.0.0",
            stages={"STAGE": StageConfig(name="STAGE")},
        )

        yaml_str = config.to_yaml()

        assert "name: yaml_cascade" in yaml_str
        assert "version: '1.0.0'" in yaml_str or "version: 1.0.0" in yaml_str

    def test_cascade_round_trip_json(self):
        """Test full round-trip serialization through JSON."""
        original = CascadeConfig(
            name="roundtrip",
            version="1.0.0",
            stages={
                "STAGE_A": StageConfig(name="STAGE_A", timeout_ms=5000),
                "STAGE_B": StageConfig(name="STAGE_B", max_retries=3),
            },
            execution_order=["STAGE_A", "STAGE_B"],
        )

        json_str = original.to_json()
        data = json.loads(json_str)
        restored = CascadeConfig.from_dict(data)

        assert restored.name == original.name
        assert restored.version == original.version
        assert len(restored.stages) == len(original.stages)
        assert restored.stages["STAGE_A"].timeout_ms == 5000
        assert restored.stages["STAGE_B"].max_retries == 3
