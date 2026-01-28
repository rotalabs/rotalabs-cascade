"""Configuration classes for cascade orchestration engine.

This module defines the configuration schema for multi-stage decision routing,
including conditions, actions, rules, and stage configurations.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConditionOperator(str, Enum):
    """Operators for condition evaluation."""

    # Comparison operators
    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="

    # Logical operators
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

    # Collection operators
    IN = "IN"
    NOT_IN = "NOT_IN"
    CONTAINS = "CONTAINS"

    # Pattern matching
    MATCHES = "MATCHES"

    # Existence operators
    EXISTS = "EXISTS"
    IS_NULL = "IS_NULL"

    # Aggregation operators
    ALL = "ALL"
    ANY = "ANY"
    NONE = "NONE"

    # Statistical operators
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT = "COUNT"


@dataclass
class Condition:
    """Represents a condition for routing decisions.

    Conditions can be simple (field-based) or composite (nested conditions).

    Attributes:
        field: Field path to evaluate (dot notation supported).
        operator: Condition operator to apply.
        value: Expected value for comparison.
        conditions: Nested conditions for logical operators (AND, OR, NOT).
    """

    field: Optional[str] = None
    operator: Optional[Union[str, ConditionOperator]] = None
    value: Optional[Any] = None
    conditions: Optional[List["Condition"]] = None

    def __post_init__(self):
        """Validate condition configuration."""
        if self.operator:
            if isinstance(self.operator, str):
                self.operator = ConditionOperator(self.operator)

        # Validate logical operators have nested conditions
        if self.operator in (ConditionOperator.AND, ConditionOperator.OR, ConditionOperator.NOT):
            if not self.conditions:
                raise ValueError(f"{self.operator} requires nested conditions")

        # Validate field-based operators have field
        elif self.operator and not self.field:
            if self.operator not in (ConditionOperator.AND, ConditionOperator.OR, ConditionOperator.NOT):
                raise ValueError(f"{self.operator} requires a field")

    def to_dict(self) -> Dict[str, Any]:
        """Convert condition to dictionary."""
        result = {}
        if self.field is not None:
            result["field"] = self.field
        if self.operator is not None:
            result["operator"] = self.operator.value if isinstance(self.operator, ConditionOperator) else self.operator
        if self.value is not None:
            result["value"] = self.value
        if self.conditions:
            result["conditions"] = [c.to_dict() for c in self.conditions]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Condition":
        """Create condition from dictionary."""
        conditions = None
        if "conditions" in data:
            conditions = [cls.from_dict(c) for c in data["conditions"]]

        return cls(
            field=data.get("field"),
            operator=data.get("operator"),
            value=data.get("value"),
            conditions=conditions,
        )


@dataclass
class RoutingAction:
    """Defines an action to take when a routing rule matches.

    Attributes:
        type: Action type (terminate, skip_to, enable_stages, disable_stages, set_field).
        target: Target stage name for skip_to actions.
        stages: List of stage names for enable/disable actions.
        field: Field path for set_field actions.
        value: Value to set for set_field actions.
    """

    type: str
    target: Optional[str] = None
    stages: Optional[List[str]] = None
    field: Optional[str] = None
    value: Optional[Any] = None

    def __post_init__(self):
        """Validate action configuration."""
        valid_types = {"terminate", "skip_to", "enable_stages", "disable_stages", "set_field"}
        if self.type not in valid_types:
            raise ValueError(f"Invalid action type: {self.type}. Must be one of {valid_types}")

        if self.type == "skip_to" and not self.target:
            raise ValueError("skip_to action requires target")
        if self.type in ("enable_stages", "disable_stages") and not self.stages:
            raise ValueError(f"{self.type} action requires stages")
        if self.type == "set_field" and not self.field:
            raise ValueError("set_field action requires field")

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        result = {"type": self.type}
        if self.target is not None:
            result["target"] = self.target
        if self.stages is not None:
            result["stages"] = self.stages
        if self.field is not None:
            result["field"] = self.field
        if self.value is not None:
            result["value"] = self.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingAction":
        """Create action from dictionary."""
        return cls(
            type=data["type"],
            target=data.get("target"),
            stages=data.get("stages"),
            field=data.get("field"),
            value=data.get("value"),
        )


@dataclass
class RoutingRule:
    """Defines a routing rule for stage execution control.

    Attributes:
        name: Unique rule identifier.
        type: Rule type (precondition, routing, postcondition).
        condition: Condition to evaluate.
        action: Action to take when condition matches.
        priority: Rule priority (higher values execute first).
    """

    name: str
    type: str
    condition: Condition
    action: RoutingAction
    priority: int = 0

    def __post_init__(self):
        """Validate rule configuration."""
        valid_types = {"precondition", "routing", "postcondition"}
        if self.type not in valid_types:
            raise ValueError(f"Invalid rule type: {self.type}. Must be one of {valid_types}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "condition": self.condition.to_dict(),
            "action": self.action.to_dict(),
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingRule":
        """Create rule from dictionary."""
        return cls(
            name=data["name"],
            type=data["type"],
            condition=Condition.from_dict(data["condition"]),
            action=RoutingAction.from_dict(data["action"]),
            priority=data.get("priority", 0),
        )


@dataclass
class StageConfig:
    """Configuration for a single execution stage.

    Attributes:
        name: Unique stage identifier.
        enabled: Whether stage is initially enabled.
        handler_type: Type of handler (determines handler resolution).
        timeout_ms: Execution timeout in milliseconds.
        max_retries: Maximum retry attempts on failure.
        retry_delay_ms: Delay between retries in milliseconds.
        can_run_parallel: Whether stage can execute in parallel with others.
        parallel_group: Group identifier for parallel execution.
        depends_on: List of stage names this stage depends on.
        routing_rules: Stage-specific routing rules.
        cache_enabled: Whether to cache stage results.
        cache_ttl_seconds: Cache TTL in seconds.
        custom_properties: Domain-specific configuration properties.
    """

    name: str
    enabled: bool = True
    handler_type: Optional[str] = None
    timeout_ms: int = 30000
    max_retries: int = 0
    retry_delay_ms: int = 1000
    can_run_parallel: bool = False
    parallel_group: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    routing_rules: List[RoutingRule] = field(default_factory=list)
    cache_enabled: bool = False
    cache_ttl_seconds: int = 3600
    custom_properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stage config to dictionary."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "handler_type": self.handler_type,
            "timeout_ms": self.timeout_ms,
            "max_retries": self.max_retries,
            "retry_delay_ms": self.retry_delay_ms,
            "can_run_parallel": self.can_run_parallel,
            "parallel_group": self.parallel_group,
            "depends_on": self.depends_on,
            "routing_rules": [r.to_dict() for r in self.routing_rules],
            "cache_enabled": self.cache_enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "custom_properties": self.custom_properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageConfig":
        """Create stage config from dictionary."""
        routing_rules = []
        if "routing_rules" in data:
            routing_rules = [RoutingRule.from_dict(r) for r in data["routing_rules"]]

        return cls(
            name=data["name"],
            enabled=data.get("enabled", True),
            handler_type=data.get("handler_type"),
            timeout_ms=data.get("timeout_ms", 30000),
            max_retries=data.get("max_retries", 0),
            retry_delay_ms=data.get("retry_delay_ms", 1000),
            can_run_parallel=data.get("can_run_parallel", False),
            parallel_group=data.get("parallel_group"),
            depends_on=data.get("depends_on", []),
            routing_rules=routing_rules,
            cache_enabled=data.get("cache_enabled", False),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 3600),
            custom_properties=data.get("custom_properties", {}),
        )


@dataclass
class CascadeConfig:
    """Complete cascade orchestration configuration.

    Attributes:
        name: Cascade configuration name.
        version: Configuration version.
        stages: Dictionary of stage configurations keyed by name.
        execution_order: Initial execution order (stage names).
        global_timeout_ms: Global execution timeout in milliseconds.
        max_parallel_stages: Maximum stages to execute in parallel.
        global_termination_conditions: Conditions that terminate entire cascade.
        enable_caching: Whether to enable result caching.
        cache_key_fields: Fields to include in cache key.
        domain_config: Domain-specific configuration.
    """

    name: str
    version: str
    stages: Dict[str, StageConfig]
    execution_order: List[str] = field(default_factory=list)
    global_timeout_ms: int = 300000
    max_parallel_stages: int = 5
    global_termination_conditions: List[Condition] = field(default_factory=list)
    enable_caching: bool = False
    cache_key_fields: List[str] = field(default_factory=list)
    domain_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate cascade configuration."""
        # Validate execution order references existing stages
        for stage_name in self.execution_order:
            if stage_name not in self.stages:
                raise ValueError(f"Execution order references unknown stage: {stage_name}")

        # Validate stage dependencies reference existing stages
        for stage in self.stages.values():
            for dep in stage.depends_on:
                if dep not in self.stages:
                    raise ValueError(f"Stage {stage.name} depends on unknown stage: {dep}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert cascade config to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()},
            "execution_order": self.execution_order,
            "global_timeout_ms": self.global_timeout_ms,
            "max_parallel_stages": self.max_parallel_stages,
            "global_termination_conditions": [c.to_dict() for c in self.global_termination_conditions],
            "enable_caching": self.enable_caching,
            "cache_key_fields": self.cache_key_fields,
            "domain_config": self.domain_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CascadeConfig":
        """Create cascade config from dictionary."""
        stages = {}
        if "stages" in data:
            stages = {name: StageConfig.from_dict(stage_data) for name, stage_data in data["stages"].items()}

        global_termination_conditions = []
        if "global_termination_conditions" in data:
            global_termination_conditions = [Condition.from_dict(c) for c in data["global_termination_conditions"]]

        return cls(
            name=data["name"],
            version=data["version"],
            stages=stages,
            execution_order=data.get("execution_order", []),
            global_timeout_ms=data.get("global_timeout_ms", 300000),
            max_parallel_stages=data.get("max_parallel_stages", 5),
            global_termination_conditions=global_termination_conditions,
            enable_caching=data.get("enable_caching", False),
            cache_key_fields=data.get("cache_key_fields", []),
            domain_config=data.get("domain_config", {}),
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "CascadeConfig":
        """Load cascade config from JSON or YAML file.

        Args:
            path: Path to configuration file (.json or .yaml/.yml).

        Returns:
            Loaded cascade configuration.

        Raises:
            ValueError: If file format is unsupported.
            ImportError: If YAML file provided but PyYAML not installed.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        with open(path, "r", encoding="utf-8") as f:
            if suffix == ".json":
                data = json.load(f)
            elif suffix in (".yaml", ".yml"):
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required to load YAML files. Install with: pip install pyyaml")
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {suffix}. Use .json, .yaml, or .yml")

        return cls.from_dict(data)

    def to_json(self, indent: int = 2) -> str:
        """Convert cascade config to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON representation of config.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        """Convert cascade config to YAML string.

        Returns:
            YAML representation of config.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required to export to YAML. Install with: pip install pyyaml")
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
