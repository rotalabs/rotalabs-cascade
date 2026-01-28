"""Rule generation from learned patterns for cascade routing.

This module generates routing rules from patterns identified by the pattern
extractor. It supports multiple rule templates and can optionally use an LLM
for generating complex rules.

The generated rules can be converted to the cascade's native RoutingRule format
or exported as YAML for review and manual editing.

Example:
    >>> from rotalabs_cascade.learning.rule_generator import RuleGenerator
    >>> from rotalabs_cascade.learning.pattern_extractor import StageFailurePattern
    >>> generator = RuleGenerator()
    >>> rule = generator.generate_from_pattern(pattern)
    >>> if rule:
    ...     routing_rule = generator.to_routing_rule(rule)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..core.config import Condition, ConditionOperator, RoutingAction, RoutingRule
from .pattern_extractor import StageFailurePattern

logger = logging.getLogger(__name__)

# Optional YAML support
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class RuleTemplate(str, Enum):
    """Types of rule templates that can be generated.

    Attributes:
        VALUE_THRESHOLD: Simple numeric comparisons (e.g., score > 0.8).
        MULTI_CONDITION: AND/OR combinations of multiple conditions.
        PATTERN_MATCH: Regex or pattern matching on string fields.
        TEMPORAL: Time-based rules (e.g., peak hours, weekends).
        BEHAVIORAL: Sequence or frequency-based patterns.
    """

    VALUE_THRESHOLD = "value_threshold"
    MULTI_CONDITION = "multi_condition"
    PATTERN_MATCH = "pattern_match"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"


# Mapping from pattern type strings to RuleTemplate
PATTERN_TYPE_TO_TEMPLATE: Dict[str, RuleTemplate] = {
    "threshold": RuleTemplate.VALUE_THRESHOLD,
    "correlation": RuleTemplate.MULTI_CONDITION,
    "reasoning": RuleTemplate.MULTI_CONDITION,
    "temporal": RuleTemplate.TEMPORAL,
    "behavioral": RuleTemplate.BEHAVIORAL,
    "custom": RuleTemplate.MULTI_CONDITION,
}


@dataclass
class GeneratedRule:
    """A routing rule generated from a learned pattern.

    This dataclass represents a machine-generated rule that can be proposed
    for human review before being deployed.

    Attributes:
        rule_id: Unique identifier for this rule.
        name: Human-readable rule name.
        description: Detailed description of what this rule does.
        template: The template type used to generate this rule.
        conditions: List of condition specifications.
        action: The action to take when rule matches (APPROVE, REJECT, ESCALATE, FLAG).
        target_stage: The stage to route to when the rule matches.
        source_pattern_id: ID of the pattern this rule was generated from.
        confidence: Confidence score from the source pattern (0.0 to 1.0).
        estimated_coverage: Estimated percentage of cases this rule would handle.
        created_at: Timestamp when this rule was generated.
    """

    rule_id: str
    name: str
    description: str
    template: RuleTemplate
    conditions: List[Dict[str, Any]]
    action: str
    target_stage: str
    source_pattern_id: str
    confidence: float
    estimated_coverage: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary representation.

        Returns:
            Dictionary representation of the generated rule.
        """
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "template": self.template.value,
            "conditions": self.conditions,
            "action": self.action,
            "target_stage": self.target_stage,
            "source_pattern_id": self.source_pattern_id,
            "confidence": self.confidence,
            "estimated_coverage": self.estimated_coverage,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedRule":
        """Create rule from dictionary representation.

        Args:
            data: Dictionary containing rule data.

        Returns:
            GeneratedRule instance.
        """
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            rule_id=data["rule_id"],
            name=data["name"],
            description=data["description"],
            template=RuleTemplate(data["template"]),
            conditions=data["conditions"],
            action=data["action"],
            target_stage=data["target_stage"],
            source_pattern_id=data["source_pattern_id"],
            confidence=data["confidence"],
            estimated_coverage=data["estimated_coverage"],
            created_at=created_at,
        )


class RuleGenerator:
    """Generates routing rules from learned patterns.

    This class analyzes patterns extracted from execution history and generates
    appropriate routing rules. It supports multiple generation strategies based
    on pattern type and can optionally use an LLM for complex rule generation.

    Attributes:
        llm_client: Optional LLM client for generating complex rules.
        min_confidence: Minimum confidence threshold for generating rules.
        min_coverage: Minimum estimated coverage for generating rules.

    Example:
        >>> generator = RuleGenerator(min_confidence=0.7)
        >>> pattern = StageFailurePattern(...)
        >>> rule = generator.generate_from_pattern(pattern)
        >>> if rule:
        ...     yaml_output = generator.to_yaml([rule])
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        min_confidence: float = 0.5,
        min_coverage: float = 0.01,
    ) -> None:
        """Initialize the rule generator.

        Args:
            llm_client: Optional LLM client for generating complex rules.
                Should have a method like `generate(prompt: str) -> str`.
            min_confidence: Minimum confidence threshold for rule generation.
            min_coverage: Minimum estimated coverage threshold.
        """
        self.llm_client = llm_client
        self.min_confidence = min_confidence
        self.min_coverage = min_coverage

        # Register generation handlers by pattern type string
        self._generators: Dict[str, Callable] = {
            "threshold": self._generate_threshold_rule,
            "correlation": self._generate_correlation_rule,
            "reasoning": self._generate_correlation_rule,
            "temporal": self._generate_temporal_rule,
            "behavioral": self._generate_behavioral_rule,
            "custom": self._generate_correlation_rule,
        }

        logger.info(
            "RuleGenerator initialized",
            extra={
                "has_llm": llm_client is not None,
                "min_confidence": min_confidence,
                "min_coverage": min_coverage,
            },
        )

    def generate_from_pattern(
        self,
        pattern: StageFailurePattern,
    ) -> Optional[GeneratedRule]:
        """Generate a routing rule from a pattern.

        This method selects the appropriate generation strategy based on
        the pattern type and attempts to create a rule.

        Args:
            pattern: The pattern to generate a rule from.

        Returns:
            A GeneratedRule if successful, None if the pattern cannot be
            converted to a rule or doesn't meet thresholds.
        """
        if pattern.confidence < self.min_confidence:
            logger.debug(
                "Pattern confidence below threshold",
                extra={
                    "pattern_id": pattern.id,
                    "confidence": pattern.confidence,
                    "threshold": self.min_confidence,
                },
            )
            return None

        # Get the appropriate generator based on pattern_type string
        generator = self._generators.get(pattern.pattern_type)

        if generator is None:
            logger.warning(
                "No generator for pattern type",
                extra={
                    "pattern_type": pattern.pattern_type,
                    "pattern_id": pattern.id,
                },
            )
            # Try LLM-based generation as fallback
            if self.llm_client is not None:
                return self._generate_llm_rule(pattern)
            return None

        try:
            rule = generator(pattern)

            if rule is not None and rule.estimated_coverage < self.min_coverage:
                logger.debug(
                    "Rule coverage below threshold",
                    extra={
                        "rule_id": rule.rule_id,
                        "coverage": rule.estimated_coverage,
                        "threshold": self.min_coverage,
                    },
                )
                return None

            if rule is not None:
                logger.info(
                    "Generated rule from pattern",
                    extra={
                        "rule_id": rule.rule_id,
                        "pattern_id": pattern.id,
                        "template": rule.template.value,
                    },
                )

            return rule

        except Exception as e:
            logger.error(
                "Failed to generate rule from pattern",
                extra={
                    "pattern_id": pattern.id,
                    "error": str(e),
                },
                exc_info=True,
            )
            return None

    def _generate_threshold_rule(
        self,
        pattern: StageFailurePattern,
    ) -> Optional[GeneratedRule]:
        """Generate a VALUE_THRESHOLD rule from a threshold pattern.

        Creates rules with simple numeric comparisons based on the
        thresholds identified in the pattern features.

        Args:
            pattern: A pattern of type "threshold".

        Returns:
            A GeneratedRule with VALUE_THRESHOLD template, or None if
            the pattern has no valid threshold features.
        """
        features = pattern.features
        if not features:
            return None

        # Extract threshold-related fields from features
        feature_path = features.get("feature_path")
        threshold_value = features.get("threshold")
        operator = features.get("operator", "gte")

        if feature_path is None or threshold_value is None:
            # Try alternative feature structure
            # Look for any numeric values that could be thresholds
            threshold_features = [
                (k, v) for k, v in features.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                and k not in ("confidence", "sample_count")
            ]
            if not threshold_features:
                return None
            feature_path, threshold_value = threshold_features[0]
            operator = "gte"

        conditions = []
        description_parts = []

        # Map operator strings to comparison operators
        op_map = {
            "gte": ">=",
            "gt": ">",
            "lte": "<=",
            "lt": "<",
            "eq": "==",
            "ne": "!=",
        }
        condition_operator = op_map.get(operator, ">=")

        condition = {
            "field": feature_path,
            "operator": condition_operator,
            "value": threshold_value,
        }
        conditions.append(condition)

        op_descriptions = {
            ">=": "greater than or equal to",
            ">": "greater than",
            "<=": "less than or equal to",
            "<": "less than",
            "==": "equal to",
            "!=": "not equal to",
        }
        op_desc = op_descriptions.get(condition_operator, "compared to")
        description_parts.append(f"{feature_path} is {op_desc} {threshold_value}")

        # Determine action based on pattern metadata
        source_stage = features.get("source_stage") or pattern.metadata.get("source_stage")
        action = self._determine_action_from_pattern(pattern)

        # Estimate coverage from pattern sample count and confidence
        estimated_coverage = self._estimate_coverage(pattern)

        rule_id = f"rule_{uuid.uuid4().hex[:8]}"
        name = f"threshold_{pattern.stage}_{feature_path.replace('.', '_')}"

        return GeneratedRule(
            rule_id=rule_id,
            name=name,
            description=f"Route when {' and '.join(description_parts)}. "
            f"Generated from pattern {pattern.id} with "
            f"{pattern.confidence:.1%} confidence based on "
            f"{pattern.sample_count} samples.",
            template=RuleTemplate.VALUE_THRESHOLD,
            conditions=conditions,
            action=action,
            target_stage=pattern.stage,
            source_pattern_id=pattern.id,
            confidence=pattern.confidence,
            estimated_coverage=estimated_coverage,
        )

    def _generate_correlation_rule(
        self,
        pattern: StageFailurePattern,
    ) -> Optional[GeneratedRule]:
        """Generate a MULTI_CONDITION rule from a correlation or reasoning pattern.

        Creates rules with AND conditions combining multiple feature checks.

        Args:
            pattern: A pattern of type "correlation", "reasoning", or "custom".

        Returns:
            A GeneratedRule with MULTI_CONDITION template, or None if
            the pattern has insufficient data.
        """
        features = pattern.features
        if not features:
            return None

        conditions = []
        description_parts = []

        # Extract feature paths from the pattern
        feature_paths = features.get("feature_paths", [])
        feature_hash = features.get("feature_hash")

        # Build conditions from available feature values
        for key, value in features.items():
            if key.startswith("val_") and isinstance(value, (str, int, float, bool)):
                # This is an actual feature value
                field_name = key[4:]  # Remove "val_" prefix
                operator = "==" if isinstance(value, (str, bool)) else ">="

                condition = {
                    "field": field_name,
                    "operator": operator,
                    "value": value,
                }
                conditions.append(condition)
                description_parts.append(f"{field_name} matches {value}")

        # If no val_ features, try to create conditions from reasoning patterns
        if not conditions and pattern.pattern_type == "reasoning":
            factors = features.get("factors", [])
            keywords = features.get("reasoning_keywords", [])

            if factors:
                for factor in factors[:3]:  # Limit to top 3 factors
                    condition = {
                        "field": f"factors.{factor}",
                        "operator": "EXISTS",
                        "value": True,
                    }
                    conditions.append(condition)
                    description_parts.append(f"factor '{factor}' present")

            if keywords and not conditions:
                # Create a pattern match condition for keywords
                keyword_pattern = "|".join(keywords[:5])
                condition = {
                    "field": "content",
                    "operator": "MATCHES",
                    "value": keyword_pattern,
                }
                conditions.append(condition)
                description_parts.append(f"content matches keywords")

        if not conditions:
            # Fallback: create a simple existence check
            if feature_paths:
                for path in feature_paths[:3]:
                    condition = {
                        "field": path,
                        "operator": "EXISTS",
                        "value": True,
                    }
                    conditions.append(condition)
                    description_parts.append(f"{path} exists")

        if not conditions:
            return None

        # Wrap multiple conditions in AND
        if len(conditions) > 1:
            combined_condition = {
                "operator": "AND",
                "conditions": conditions,
            }
            conditions = [combined_condition]

        action = self._determine_action_from_pattern(pattern)
        estimated_coverage = self._estimate_coverage(pattern)

        rule_id = f"rule_{uuid.uuid4().hex[:8]}"
        name = f"{pattern.pattern_type}_{pattern.stage}_{pattern.id[:8]}"

        return GeneratedRule(
            rule_id=rule_id,
            name=name,
            description=f"Route when {', '.join(description_parts)}. "
            f"Generated from {pattern.pattern_type} pattern {pattern.id} with "
            f"{pattern.confidence:.1%} confidence.",
            template=RuleTemplate.MULTI_CONDITION,
            conditions=conditions,
            action=action,
            target_stage=pattern.stage,
            source_pattern_id=pattern.id,
            confidence=pattern.confidence,
            estimated_coverage=estimated_coverage,
        )

    def _generate_temporal_rule(
        self,
        pattern: StageFailurePattern,
    ) -> Optional[GeneratedRule]:
        """Generate a TEMPORAL rule from a temporal pattern.

        Creates time-based rules from patterns that identify specific
        time characteristics.

        Args:
            pattern: A pattern of type "temporal".

        Returns:
            A GeneratedRule with TEMPORAL template, or None if the
            pattern lacks temporal data.
        """
        features = pattern.features
        if not features:
            return None

        conditions = []
        description_parts = []

        # Extract timing information
        stage_time_ms = features.get("stage_time_ms")
        total_elapsed_ms = features.get("total_elapsed_ms")
        timestamp_fields = features.get("timestamp_fields", [])

        # Create conditions based on timing patterns
        if stage_time_ms is not None:
            # Create a condition for processing time threshold
            condition = {
                "field": "processing_time_ms",
                "operator": ">=",
                "value": stage_time_ms,
            }
            conditions.append(condition)
            description_parts.append(f"processing time >= {stage_time_ms}ms")

        if total_elapsed_ms is not None and not conditions:
            condition = {
                "field": "elapsed_time_ms",
                "operator": ">=",
                "value": total_elapsed_ms,
            }
            conditions.append(condition)
            description_parts.append(f"total elapsed time >= {total_elapsed_ms}ms")

        # Check for timestamp-based patterns in metadata
        temporal_window = pattern.metadata.get("temporal_window")
        temporal_feature = pattern.metadata.get("temporal_feature", "hour")

        if temporal_window is not None:
            if temporal_feature == "hour":
                window_center = float(temporal_window)
                window_start = int(window_center - 2) % 24
                window_end = int(window_center + 2) % 24

                if window_start < window_end:
                    condition = {
                        "operator": "AND",
                        "conditions": [
                            {"field": "hour", "operator": ">=", "value": window_start},
                            {"field": "hour", "operator": "<=", "value": window_end},
                        ],
                    }
                    description_parts.append(
                        f"time between {window_start}:00 and {window_end}:00"
                    )
                else:
                    condition = {
                        "operator": "OR",
                        "conditions": [
                            {"field": "hour", "operator": ">=", "value": window_start},
                            {"field": "hour", "operator": "<=", "value": window_end},
                        ],
                    }
                    description_parts.append(
                        f"time between {window_start}:00 and {window_end}:00 (overnight)"
                    )
                conditions.append(condition)

        if not conditions:
            return None

        action = self._determine_action_from_pattern(pattern)
        estimated_coverage = self._estimate_coverage(pattern)

        rule_id = f"rule_{uuid.uuid4().hex[:8]}"
        name = f"temporal_{pattern.stage}_{pattern.id[:8]}"

        return GeneratedRule(
            rule_id=rule_id,
            name=name,
            description=f"Route based on timing: {', '.join(description_parts)}. "
            f"Generated from temporal pattern {pattern.id} with "
            f"{pattern.confidence:.1%} confidence.",
            template=RuleTemplate.TEMPORAL,
            conditions=conditions,
            action=action,
            target_stage=pattern.stage,
            source_pattern_id=pattern.id,
            confidence=pattern.confidence,
            estimated_coverage=estimated_coverage,
        )

    def _generate_behavioral_rule(
        self,
        pattern: StageFailurePattern,
    ) -> Optional[GeneratedRule]:
        """Generate a BEHAVIORAL rule from a behavioral pattern.

        Creates rules based on behavioral patterns like execution flow
        characteristics and routing decisions.

        Args:
            pattern: A pattern of type "behavioral".

        Returns:
            A GeneratedRule with BEHAVIORAL template, or None if the
            pattern lacks behavioral data.
        """
        features = pattern.features
        if not features:
            return None

        conditions = []
        description_parts = []

        # Extract behavioral features
        stages_executed = features.get("stages_executed", [])
        stage_count = features.get("stage_count")
        routing_count = features.get("routing_count")
        decision_types = features.get("decision_types", [])
        avg_confidence = features.get("avg_confidence")
        max_confidence = features.get("max_confidence")

        # Create conditions based on execution patterns
        if stage_count is not None:
            condition = {
                "field": "stage_count",
                "operator": ">=",
                "value": stage_count,
            }
            conditions.append(condition)
            description_parts.append(f"at least {stage_count} stages executed")

        if avg_confidence is not None:
            condition = {
                "field": "avg_confidence",
                "operator": ">=",
                "value": avg_confidence,
            }
            conditions.append(condition)
            description_parts.append(f"average confidence >= {avg_confidence:.2f}")

        if decision_types:
            for dt in decision_types[:2]:
                condition = {
                    "field": f"decision_type.{dt}",
                    "operator": "EXISTS",
                    "value": True,
                }
                conditions.append(condition)
                description_parts.append(f"decision type '{dt}' present")

        if stages_executed and not conditions:
            # Check for specific stage execution
            for stage in stages_executed[:3]:
                condition = {
                    "field": f"executed_stages.{stage}",
                    "operator": "EXISTS",
                    "value": True,
                }
                conditions.append(condition)
                description_parts.append(f"stage '{stage}' executed")

        if not conditions:
            return None

        # Wrap multiple conditions in AND
        if len(conditions) > 1:
            combined_condition = {
                "operator": "AND",
                "conditions": conditions,
            }
            conditions = [combined_condition]

        action = self._determine_action_from_pattern(pattern)
        estimated_coverage = self._estimate_coverage(pattern)

        rule_id = f"rule_{uuid.uuid4().hex[:8]}"
        name = f"behavioral_{pattern.stage}_{pattern.id[:8]}"

        return GeneratedRule(
            rule_id=rule_id,
            name=name,
            description=f"Route based on behavior: {', '.join(description_parts)}. "
            f"Generated from behavioral pattern {pattern.id} with "
            f"{pattern.confidence:.1%} confidence.",
            template=RuleTemplate.BEHAVIORAL,
            conditions=conditions,
            action=action,
            target_stage=pattern.stage,
            source_pattern_id=pattern.id,
            confidence=pattern.confidence,
            estimated_coverage=estimated_coverage,
        )

    def _generate_llm_rule(
        self,
        pattern: StageFailurePattern,
    ) -> Optional[GeneratedRule]:
        """Generate a rule using an LLM for complex patterns.

        Falls back to LLM-based generation when standard generators
        cannot handle a pattern. Gracefully degrades if no LLM client
        is available.

        Args:
            pattern: The pattern to generate a rule from.

        Returns:
            A GeneratedRule if LLM generation succeeds, None otherwise.
        """
        if self.llm_client is None:
            logger.debug(
                "LLM client not available for complex pattern",
                extra={"pattern_id": pattern.id},
            )
            return None

        try:
            # Build prompt for LLM
            prompt = self._build_llm_prompt(pattern)

            # Call LLM (assumes a simple generate interface)
            if hasattr(self.llm_client, "generate"):
                response = self.llm_client.generate(prompt)
            elif hasattr(self.llm_client, "complete"):
                response = self.llm_client.complete(prompt)
            elif callable(self.llm_client):
                response = self.llm_client(prompt)
            else:
                logger.warning(
                    "LLM client does not have a recognized generation method",
                    extra={"client_type": type(self.llm_client).__name__},
                )
                return None

            # Parse LLM response
            rule = self._parse_llm_response(response, pattern)

            if rule is not None:
                logger.info(
                    "Generated rule using LLM",
                    extra={
                        "rule_id": rule.rule_id,
                        "pattern_id": pattern.id,
                    },
                )

            return rule

        except Exception as e:
            logger.error(
                "LLM rule generation failed",
                extra={
                    "pattern_id": pattern.id,
                    "error": str(e),
                },
                exc_info=True,
            )
            return None

    def _build_llm_prompt(self, pattern: StageFailurePattern) -> str:
        """Build a prompt for LLM-based rule generation.

        Args:
            pattern: The pattern to build a prompt for.

        Returns:
            A prompt string for the LLM.
        """
        features_str = json.dumps(pattern.features, indent=2, default=str)
        metadata_str = json.dumps(pattern.metadata, indent=2, default=str)

        return f"""Generate a routing rule for the following pattern:

Pattern ID: {pattern.id}
Pattern Type: {pattern.pattern_type}
Stage: {pattern.stage}
Confidence: {pattern.confidence:.1%}
Sample Count: {pattern.sample_count}
Features: {features_str}
Metadata: {metadata_str}

Please provide:
1. A descriptive name for the rule
2. A clear description of what the rule does
3. The conditions in JSON format (with fields: field, operator, value)
4. The recommended action (APPROVE, REJECT, ESCALATE, or FLAG)

Format your response as JSON with keys: name, description, conditions, action
"""

    def _parse_llm_response(
        self,
        response: str,
        pattern: StageFailurePattern,
    ) -> Optional[GeneratedRule]:
        """Parse an LLM response into a GeneratedRule.

        Args:
            response: The raw LLM response string.
            pattern: The source pattern.

        Returns:
            A GeneratedRule if parsing succeeds, None otherwise.
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                logger.warning(
                    "Could not find JSON in LLM response",
                    extra={"response_preview": response[:200]},
                )
                return None

            name = data.get("name", f"llm_rule_{pattern.stage}")
            description = data.get("description", "LLM-generated rule")
            conditions = data.get("conditions", [])
            action = data.get("action", "FLAG")

            # Validate action
            if action not in ("APPROVE", "REJECT", "ESCALATE", "FLAG"):
                action = "FLAG"

            # Ensure conditions is a list
            if isinstance(conditions, dict):
                conditions = [conditions]

            rule_id = f"rule_{uuid.uuid4().hex[:8]}"

            return GeneratedRule(
                rule_id=rule_id,
                name=name,
                description=description,
                template=RuleTemplate.MULTI_CONDITION,
                conditions=conditions,
                action=action,
                target_stage=pattern.stage,
                source_pattern_id=pattern.id,
                confidence=pattern.confidence,
                estimated_coverage=self._estimate_coverage(pattern),
            )

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse LLM response as JSON",
                extra={"error": str(e)},
            )
            return None

    def _determine_action_from_pattern(self, pattern: StageFailurePattern) -> str:
        """Determine the appropriate action based on pattern characteristics.

        Args:
            pattern: The source pattern.

        Returns:
            Action string (APPROVE, REJECT, ESCALATE, or FLAG).
        """
        # Check metadata for action hints
        if pattern.metadata:
            if pattern.metadata.get("is_rejection") or pattern.metadata.get("blocked"):
                return "REJECT"
            if pattern.metadata.get("needs_review") or pattern.metadata.get("escalate"):
                return "ESCALATE"
            if pattern.metadata.get("approved") or pattern.metadata.get("passed"):
                return "APPROVE"

        # Default based on confidence
        if pattern.confidence >= 0.9:
            return "APPROVE"  # High confidence - can route directly
        elif pattern.confidence >= 0.7:
            return "FLAG"  # Medium confidence - flag for review
        else:
            return "ESCALATE"  # Lower confidence - escalate

    def _estimate_coverage(self, pattern: StageFailurePattern) -> float:
        """Estimate the coverage of a rule based on pattern statistics.

        Args:
            pattern: The source pattern.

        Returns:
            Estimated coverage as a fraction (0.0 to 1.0).
        """
        # Base coverage estimate from confidence and sample count
        # More samples with higher confidence = higher coverage estimate
        base_coverage = pattern.confidence * 0.1

        # Adjust for sample count (more samples = more reliable coverage estimate)
        sample_factor = min(pattern.sample_count / 100, 1.0)
        adjusted_coverage = base_coverage * (0.5 + 0.5 * sample_factor)

        return min(adjusted_coverage, 1.0)

    def to_routing_rule(self, generated: GeneratedRule) -> RoutingRule:
        """Convert a GeneratedRule to cascade's RoutingRule format.

        This method transforms the generated rule into the native
        RoutingRule format used by the cascade engine.

        Args:
            generated: The GeneratedRule to convert.

        Returns:
            A RoutingRule ready for use in cascade configuration.
        """
        # Convert conditions to Condition objects
        condition = self._convert_conditions(generated.conditions)

        # Map action string to RoutingAction
        action_type, action_target = self._map_action(
            generated.action, generated.target_stage
        )

        routing_action = RoutingAction(
            type=action_type,
            target=action_target if action_type == "skip_to" else None,
        )

        return RoutingRule(
            name=generated.name,
            type="routing",
            condition=condition,
            action=routing_action,
            priority=int(generated.confidence * 100),
        )

    def _convert_conditions(
        self,
        conditions: List[Dict[str, Any]],
    ) -> Condition:
        """Convert condition dictionaries to Condition objects.

        Args:
            conditions: List of condition dictionaries.

        Returns:
            A Condition object (possibly nested).
        """
        if not conditions:
            # Return a condition that always matches
            return Condition(
                field="_always",
                operator=ConditionOperator.EXISTS,
            )

        if len(conditions) == 1:
            return self._dict_to_condition(conditions[0])

        # Multiple conditions - wrap in AND
        nested_conditions = [
            self._dict_to_condition(c) for c in conditions
        ]

        return Condition(
            operator=ConditionOperator.AND,
            conditions=nested_conditions,
        )

    def _dict_to_condition(self, cond_dict: Dict[str, Any]) -> Condition:
        """Convert a single condition dictionary to a Condition object.

        Args:
            cond_dict: Dictionary with condition specification.

        Returns:
            A Condition object.
        """
        # Handle nested conditions (AND/OR)
        if "conditions" in cond_dict:
            operator_str = cond_dict.get("operator", "AND")
            nested = [
                self._dict_to_condition(c) for c in cond_dict["conditions"]
            ]
            return Condition(
                operator=ConditionOperator(operator_str),
                conditions=nested,
            )

        # Handle simple field conditions
        field = cond_dict.get("field")
        operator_str = cond_dict.get("operator", "==")
        value = cond_dict.get("value")

        # Map operator strings to ConditionOperator
        operator_map = {
            "==": ConditionOperator.EQ,
            "!=": ConditionOperator.NE,
            ">": ConditionOperator.GT,
            ">=": ConditionOperator.GE,
            "<": ConditionOperator.LT,
            "<=": ConditionOperator.LE,
            "IN": ConditionOperator.IN,
            "NOT_IN": ConditionOperator.NOT_IN,
            "CONTAINS": ConditionOperator.CONTAINS,
            "MATCHES": ConditionOperator.MATCHES,
            "EXISTS": ConditionOperator.EXISTS,
            "IS_NULL": ConditionOperator.IS_NULL,
        }

        operator = operator_map.get(operator_str, ConditionOperator.EQ)

        return Condition(
            field=field,
            operator=operator,
            value=value,
        )

    def _map_action(
        self,
        action: str,
        target_stage: str,
    ) -> tuple:
        """Map generated rule action to RoutingAction type and target.

        Args:
            action: The action string (APPROVE, REJECT, ESCALATE, FLAG).
            target_stage: The target stage name.

        Returns:
            Tuple of (action_type, action_target).
        """
        action_mapping = {
            "APPROVE": ("skip_to", target_stage),
            "REJECT": ("terminate", None),
            "ESCALATE": ("skip_to", "escalation"),
            "FLAG": ("set_field", None),
        }

        return action_mapping.get(action, ("skip_to", target_stage))

    def to_yaml(self, rules: List[GeneratedRule]) -> str:
        """Export generated rules as YAML.

        This method exports rules in a human-readable YAML format
        suitable for review and manual editing.

        Args:
            rules: List of GeneratedRule objects to export.

        Returns:
            YAML string representation of the rules.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to export to YAML. "
                "Install with: pip install pyyaml"
            )

        export_data = {
            "generated_rules": [rule.to_dict() for rule in rules],
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "rule_count": len(rules),
                "templates_used": list(set(r.template.value for r in rules)),
            },
        }

        return yaml.dump(
            export_data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    def generate_batch(
        self,
        patterns: List[StageFailurePattern],
    ) -> List[GeneratedRule]:
        """Generate rules from a batch of patterns.

        Convenience method for processing multiple patterns at once.

        Args:
            patterns: List of patterns to generate rules from.

        Returns:
            List of successfully generated rules.
        """
        rules = []

        for pattern in patterns:
            rule = self.generate_from_pattern(pattern)
            if rule is not None:
                rules.append(rule)

        logger.info(
            "Batch rule generation complete",
            extra={
                "patterns_processed": len(patterns),
                "rules_generated": len(rules),
            },
        )

        return rules
