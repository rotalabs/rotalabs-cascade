"""Pattern extraction from cascade stage failures for optimization.

This module extracts patterns from cascade stage failures to identify
opportunities for optimization. When expensive stages (like AI) consistently
catch certain patterns, those patterns can potentially be learned and moved
to cheaper stages (like rules).

The pattern learning system is domain-agnostic and works with any cascade
configuration.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..core.context import ExecutionContext, StageResult

logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    """Types of patterns that can be extracted from stage failures.

    This enum is provided for backward compatibility with other modules
    that may reference pattern types as an enum.

    Attributes:
        THRESHOLD: Simple threshold-based patterns (e.g., value > X).
        CORRELATION: Feature correlation patterns.
        REASONING: Patterns extracted from AI reasoning/explanations.
        TEMPORAL: Time-based patterns.
        BEHAVIORAL: Execution flow and behavioral patterns.
        SEQUENCE: Sequential behavior patterns.
        FREQUENCY: Frequency-based patterns.
    """

    THRESHOLD = "threshold"
    CORRELATION = "correlation"
    REASONING = "reasoning"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    SEQUENCE = "sequence"
    FREQUENCY = "frequency"


@dataclass
class PatternConfig:
    """Configuration for pattern extraction.

    Attributes:
        min_confidence: Minimum confidence threshold for pattern extraction.
        min_samples_for_candidate: Minimum samples before considering migration.
        max_patterns: Maximum number of patterns to track (LRU eviction).
        feature_extractors: Custom feature extractors by stage type.
        pattern_ttl_hours: Time-to-live for patterns without new samples.
        enable_threshold_extraction: Enable threshold pattern extraction.
        enable_correlation_extraction: Enable correlation pattern extraction.
        enable_reasoning_extraction: Enable reasoning pattern extraction.
        enable_temporal_extraction: Enable temporal pattern extraction.
        enable_behavioral_extraction: Enable behavioral pattern extraction.
    """

    min_confidence: float = 0.7
    min_samples_for_candidate: int = 10
    max_patterns: int = 1000
    feature_extractors: Dict[str, Callable[[ExecutionContext, StageResult], Dict[str, Any]]] = field(
        default_factory=dict
    )
    pattern_ttl_hours: int = 168  # 7 days
    enable_threshold_extraction: bool = True
    enable_correlation_extraction: bool = True
    enable_reasoning_extraction: bool = True
    enable_temporal_extraction: bool = True
    enable_behavioral_extraction: bool = True


@dataclass
class StageFailurePattern:
    """Represents a pattern extracted from stage failures.

    Patterns capture recurring characteristics in data that cause specific
    stages to trigger. By identifying these patterns, we can potentially
    move detection to cheaper, earlier stages.

    Attributes:
        id: Unique identifier for this pattern.
        stage: Name of the cascade stage where this pattern was detected.
        pattern_type: Type of pattern (threshold, correlation, reasoning,
            temporal, behavioral).
        features: Extracted feature values that define this pattern.
        confidence: Confidence score (0-1) in this pattern's reliability.
        sample_count: Number of samples that matched this pattern.
        first_seen: Timestamp when pattern was first observed.
        last_seen: Timestamp when pattern was last observed.
        metadata: Additional pattern metadata.
    """

    id: str
    stage: str
    pattern_type: str
    features: Dict[str, Any]
    confidence: float
    sample_count: int
    first_seen: datetime
    last_seen: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary representation."""
        return {
            "id": self.id,
            "stage": self.stage,
            "pattern_type": self.pattern_type,
            "features": self.features,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageFailurePattern":
        """Create pattern from dictionary representation."""
        return cls(
            id=data["id"],
            stage=data["stage"],
            pattern_type=data["pattern_type"],
            features=data["features"],
            confidence=data["confidence"],
            sample_count=data["sample_count"],
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PatternLearningInsight:
    """Aggregated insights from pattern learning.

    Provides a summary of learned patterns and recommendations for
    optimizing the cascade configuration.

    Attributes:
        patterns: List of all extracted patterns.
        migration_candidates: Pattern IDs that could move to cheaper stages.
        estimated_cost_reduction: Estimated percentage cost reduction if
            candidates are migrated.
        detection_rate_impact: Estimated impact on detection rate (negative
            means potential missed detections).
    """

    patterns: List[StageFailurePattern]
    migration_candidates: List[str]
    estimated_cost_reduction: float
    detection_rate_impact: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert insights to dictionary representation."""
        return {
            "patterns": [p.to_dict() for p in self.patterns],
            "migration_candidates": self.migration_candidates,
            "estimated_cost_reduction": self.estimated_cost_reduction,
            "detection_rate_impact": self.detection_rate_impact,
            "total_patterns": len(self.patterns),
            "candidate_count": len(self.migration_candidates),
        }


class PatternExtractor:
    """Extracts patterns from cascade stage failures for optimization.

    The PatternExtractor analyzes stage results to identify recurring
    patterns that could potentially be moved to cheaper stages. For example,
    if an AI stage consistently catches a specific type of input based on
    certain feature combinations, that pattern could be encoded as a rule.

    Example:
        >>> extractor = PatternExtractor()
        >>> pattern = extractor.learn_from_failure(context, "ai_analysis", result)
        >>> if pattern:
        ...     print(f"Learned pattern: {pattern.id}")
        >>> candidates = extractor.get_migration_candidates(min_confidence=0.9)
        >>> insights = extractor.get_insights()
    """

    def __init__(self, config: Optional[PatternConfig] = None):
        """Initialize the pattern extractor.

        Args:
            config: Optional configuration for pattern extraction.
        """
        self._config = config or PatternConfig()
        self._patterns: Dict[str, StageFailurePattern] = {}
        self._pattern_access_order: List[str] = []  # For LRU eviction
        self._feature_hashes: Dict[str, str] = {}  # Hash to pattern ID mapping

    def learn_from_failure(
        self,
        context: ExecutionContext,
        stage: str,
        result: StageResult,
    ) -> Optional[StageFailurePattern]:
        """Learn patterns from a stage failure or detection.

        Analyzes the execution context and stage result to extract
        patterns that characterize this type of detection.

        Args:
            context: The execution context containing input data.
            stage: Name of the stage that detected/flagged the input.
            result: The stage result containing detection details.

        Returns:
            The extracted or updated pattern, or None if no pattern found.
        """
        # Extract patterns based on different strategies
        extracted_patterns: List[Dict[str, Any]] = []

        if self._config.enable_threshold_extraction:
            threshold_patterns = self._extract_threshold_patterns(context, stage, result)
            extracted_patterns.extend(threshold_patterns)

        if self._config.enable_correlation_extraction:
            correlation_patterns = self._extract_correlation_patterns(context, stage, result)
            extracted_patterns.extend(correlation_patterns)

        if self._config.enable_reasoning_extraction:
            reasoning_patterns = self._extract_reasoning_patterns(context, stage, result)
            extracted_patterns.extend(reasoning_patterns)

        if self._config.enable_temporal_extraction:
            temporal_patterns = self._extract_temporal_patterns(context, stage, result)
            extracted_patterns.extend(temporal_patterns)

        if self._config.enable_behavioral_extraction:
            behavioral_patterns = self._extract_behavioral_patterns(context, stage, result)
            extracted_patterns.extend(behavioral_patterns)

        # Process custom extractors if configured
        if stage in self._config.feature_extractors:
            try:
                custom_features = self._config.feature_extractors[stage](context, result)
                if custom_features:
                    extracted_patterns.append({
                        "type": "custom",
                        "features": custom_features,
                        "confidence": 0.8,
                    })
            except Exception as e:
                logger.warning(f"Custom feature extractor failed for stage {stage}: {e}")

        if not extracted_patterns:
            return None

        # Select the best pattern (highest confidence)
        best_pattern = max(extracted_patterns, key=lambda p: p.get("confidence", 0))

        # Create or update pattern
        return self._update_or_create_pattern(
            stage=stage,
            pattern_type=best_pattern["type"],
            features=best_pattern["features"],
            confidence=best_pattern.get("confidence", self._config.min_confidence),
            metadata=best_pattern.get("metadata", {}),
        )

    def _extract_threshold_patterns(
        self,
        context: ExecutionContext,
        stage: str,
        result: StageResult,
    ) -> List[Dict[str, Any]]:
        """Extract threshold-based patterns for rules stage migration.

        Identifies numeric features that exceed certain thresholds,
        which could be encoded as simple comparison rules.

        Args:
            context: The execution context.
            stage: The stage name.
            result: The stage result.

        Returns:
            List of extracted threshold patterns.
        """
        patterns = []
        data = context.data

        # Extract numeric features from the data
        numeric_features = self._extract_numeric_features(data)

        if not numeric_features:
            return patterns

        # Look for extreme values that might indicate threshold opportunities
        for feature_path, value in numeric_features.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Check if this is an unusually high or low value
                # In a real implementation, this would use historical statistics
                pattern_features = {
                    "feature_path": feature_path,
                    "value": value,
                    "operator": "gte" if value > 0 else "lte",
                    "threshold": value,
                }

                patterns.append({
                    "type": "threshold",
                    "features": pattern_features,
                    "confidence": 0.6,  # Base confidence for threshold patterns
                    "metadata": {"source_stage": stage},
                })

        return patterns

    def _extract_correlation_patterns(
        self,
        context: ExecutionContext,
        stage: str,
        result: StageResult,
    ) -> List[Dict[str, Any]]:
        """Extract correlation patterns for ML stage migration.

        Identifies feature combinations that correlate with detections,
        which could be encoded as ML model features or multi-condition rules.

        Args:
            context: The execution context.
            stage: The stage name.
            result: The stage result.

        Returns:
            List of extracted correlation patterns.
        """
        patterns = []
        data = context.data

        # Extract feature combinations
        feature_values = self._extract_all_features(data)

        if len(feature_values) < 2:
            return patterns

        # Create a correlation pattern from the feature combination
        pattern_features = {
            "feature_count": len(feature_values),
            "feature_paths": list(feature_values.keys())[:10],  # Limit to 10 features
            "feature_hash": self._hash_features(feature_values),
        }

        # Include actual values for top features (by name sorting for consistency)
        sorted_features = sorted(feature_values.items())[:5]
        for path, value in sorted_features:
            if isinstance(value, (str, int, float, bool)):
                pattern_features[f"val_{path}"] = value

        patterns.append({
            "type": "correlation",
            "features": pattern_features,
            "confidence": 0.65,
            "metadata": {"source_stage": stage, "total_features": len(feature_values)},
        })

        return patterns

    def _extract_reasoning_patterns(
        self,
        context: ExecutionContext,
        stage: str,
        result: StageResult,
    ) -> List[Dict[str, Any]]:
        """Extract reasoning patterns from AI stage outputs.

        Analyzes AI stage explanations and reasoning to identify
        patterns that could be encoded as explicit rules.

        Args:
            context: The execution context.
            stage: The stage name.
            result: The stage result.

        Returns:
            List of extracted reasoning patterns.
        """
        patterns = []

        # Check if the result contains reasoning data
        result_data = result.data or {}
        reasoning = result_data.get("reasoning") or result_data.get("explanation")
        factors = result_data.get("factors") or result_data.get("signals") or []

        if not reasoning and not factors:
            return patterns

        pattern_features: Dict[str, Any] = {}

        # Extract key factors if available
        if factors:
            if isinstance(factors, list):
                pattern_features["factor_count"] = len(factors)
                pattern_features["factors"] = factors[:5]  # Top 5 factors
            elif isinstance(factors, dict):
                pattern_features["factor_count"] = len(factors)
                pattern_features["factors"] = list(factors.keys())[:5]

        # Extract keywords from reasoning if available
        if reasoning and isinstance(reasoning, str):
            # Simple keyword extraction (domain-agnostic)
            words = reasoning.lower().split()
            keywords = [w for w in words if len(w) > 4 and w.isalpha()][:10]
            if keywords:
                pattern_features["reasoning_keywords"] = keywords
                pattern_features["reasoning_length"] = len(reasoning)

        if pattern_features:
            patterns.append({
                "type": "reasoning",
                "features": pattern_features,
                "confidence": 0.75,  # Higher confidence for AI-derived patterns
                "metadata": {
                    "source_stage": stage,
                    "has_reasoning": bool(reasoning),
                    "has_factors": bool(factors),
                },
            })

        return patterns

    def _extract_temporal_patterns(
        self,
        context: ExecutionContext,
        stage: str,
        result: StageResult,
    ) -> List[Dict[str, Any]]:
        """Extract temporal patterns from execution timing.

        Identifies time-based patterns that could indicate
        specific processing characteristics.

        Args:
            context: The execution context.
            stage: The stage name.
            result: The stage result.

        Returns:
            List of extracted temporal patterns.
        """
        patterns = []

        # Extract timing information
        pattern_features: Dict[str, Any] = {
            "stage_time_ms": result.time_ms,
        }

        # Check for timestamp fields in data
        data = context.data
        timestamp_fields = self._find_timestamp_fields(data)
        if timestamp_fields:
            pattern_features["timestamp_fields"] = list(timestamp_fields.keys())[:5]

        # Add execution timing context
        pattern_features["total_elapsed_ms"] = context.elapsed_ms

        if pattern_features:
            patterns.append({
                "type": "temporal",
                "features": pattern_features,
                "confidence": 0.5,  # Lower confidence for temporal patterns
                "metadata": {"source_stage": stage},
            })

        return patterns

    def _extract_behavioral_patterns(
        self,
        context: ExecutionContext,
        stage: str,
        result: StageResult,
    ) -> List[Dict[str, Any]]:
        """Extract behavioral patterns from execution flow.

        Identifies patterns in how the cascade executed, including
        routing decisions and stage interactions.

        Args:
            context: The execution context.
            stage: The stage name.
            result: The stage result.

        Returns:
            List of extracted behavioral patterns.
        """
        patterns = []

        # Analyze the execution path
        execution_result = context.get_result()
        stage_results = execution_result.get("stage_results", {})
        routing_decisions = execution_result.get("routing_decisions", [])

        pattern_features: Dict[str, Any] = {
            "stages_executed": list(stage_results.keys()),
            "stage_count": len(stage_results),
        }

        # Analyze routing patterns
        if routing_decisions:
            pattern_features["routing_count"] = len(routing_decisions)
            decision_types = [d.get("type") for d in routing_decisions if d.get("type")]
            if decision_types:
                pattern_features["decision_types"] = list(set(decision_types))

        # Check confidence patterns across stages
        confidences = []
        for stage_name, stage_data in stage_results.items():
            if isinstance(stage_data, dict) and "confidence" in stage_data:
                confidences.append(stage_data["confidence"])

        if confidences:
            pattern_features["avg_confidence"] = sum(confidences) / len(confidences)
            pattern_features["max_confidence"] = max(confidences)

        if len(pattern_features) > 2:  # More than just stages_executed and stage_count
            patterns.append({
                "type": "behavioral",
                "features": pattern_features,
                "confidence": 0.55,
                "metadata": {"source_stage": stage},
            })

        return patterns

    def _update_or_create_pattern(
        self,
        stage: str,
        pattern_type: str,
        features: Dict[str, Any],
        confidence: float,
        metadata: Dict[str, Any],
    ) -> StageFailurePattern:
        """Update an existing pattern or create a new one.

        Args:
            stage: The stage name.
            pattern_type: Type of pattern.
            features: Pattern features.
            confidence: Confidence score.
            metadata: Additional metadata.

        Returns:
            The created or updated pattern.
        """
        # Generate a hash for the feature combination
        feature_hash = self._hash_features(features)
        pattern_key = f"{stage}:{pattern_type}:{feature_hash}"

        now = datetime.now()

        if pattern_key in self._feature_hashes:
            # Update existing pattern
            pattern_id = self._feature_hashes[pattern_key]
            pattern = self._patterns[pattern_id]

            # Update pattern statistics
            pattern.sample_count += 1
            pattern.last_seen = now

            # Update confidence with exponential moving average
            alpha = 0.1
            pattern.confidence = alpha * confidence + (1 - alpha) * pattern.confidence

            # Update LRU order
            if pattern_id in self._pattern_access_order:
                self._pattern_access_order.remove(pattern_id)
            self._pattern_access_order.append(pattern_id)

            logger.debug(
                f"Updated pattern {pattern_id}: samples={pattern.sample_count}, "
                f"confidence={pattern.confidence:.3f}"
            )

            return pattern

        # Create new pattern
        pattern_id = str(uuid.uuid4())[:8]
        pattern = StageFailurePattern(
            id=pattern_id,
            stage=stage,
            pattern_type=pattern_type,
            features=features,
            confidence=confidence,
            sample_count=1,
            first_seen=now,
            last_seen=now,
            metadata=metadata,
        )

        # Store pattern
        self._patterns[pattern_id] = pattern
        self._feature_hashes[pattern_key] = pattern_id
        self._pattern_access_order.append(pattern_id)

        # Enforce max patterns limit (LRU eviction)
        self._enforce_max_patterns()

        logger.debug(f"Created new pattern {pattern_id}: type={pattern_type}, stage={stage}")

        return pattern

    def _enforce_max_patterns(self) -> None:
        """Enforce maximum pattern limit using LRU eviction."""
        while len(self._patterns) > self._config.max_patterns:
            # Remove least recently used pattern
            oldest_id = self._pattern_access_order.pop(0)
            pattern = self._patterns.pop(oldest_id, None)

            if pattern:
                # Remove from feature hash mapping
                feature_hash = self._hash_features(pattern.features)
                pattern_key = f"{pattern.stage}:{pattern.pattern_type}:{feature_hash}"
                self._feature_hashes.pop(pattern_key, None)

                logger.debug(f"Evicted pattern {oldest_id} (LRU)")

    def get_migration_candidates(
        self,
        min_confidence: float = 0.8,
        min_samples: int = 10,
    ) -> List[StageFailurePattern]:
        """Get patterns that are candidates for migration to cheaper stages.

        Returns patterns that have high confidence and sufficient samples
        to warrant consideration for migration.

        Args:
            min_confidence: Minimum confidence threshold.
            min_samples: Minimum number of samples.

        Returns:
            List of patterns suitable for migration.
        """
        candidates = []

        for pattern in self._patterns.values():
            if pattern.confidence >= min_confidence and pattern.sample_count >= min_samples:
                candidates.append(pattern)

        # Sort by confidence * sample_count (impact score)
        candidates.sort(
            key=lambda p: p.confidence * p.sample_count,
            reverse=True,
        )

        logger.info(
            f"Found {len(candidates)} migration candidates "
            f"(min_confidence={min_confidence}, min_samples={min_samples})"
        )

        return candidates

    def get_insights(self) -> PatternLearningInsight:
        """Get aggregated learning insights.

        Computes overall statistics and recommendations from
        all learned patterns.

        Returns:
            PatternLearningInsight with analysis and recommendations.
        """
        patterns = list(self._patterns.values())
        candidates = self.get_migration_candidates(
            min_confidence=self._config.min_confidence,
            min_samples=self._config.min_samples_for_candidate,
        )

        # Estimate cost reduction based on candidate patterns
        # This is a simplified estimation - real implementation would
        # use actual cost metrics from the cascade configuration
        total_samples = sum(p.sample_count for p in patterns) if patterns else 1
        candidate_samples = sum(p.sample_count for p in candidates)

        # Assume migrating to cheaper stage saves ~80% of stage cost
        estimated_cost_reduction = (candidate_samples / total_samples) * 0.8 if total_samples > 0 else 0.0

        # Estimate detection rate impact based on confidence
        # Lower confidence patterns have higher risk of missed detections
        if candidates:
            avg_confidence = sum(p.confidence for p in candidates) / len(candidates)
            # Impact is negative proportional to uncertainty (1 - confidence)
            detection_rate_impact = -(1 - avg_confidence) * (candidate_samples / total_samples)
        else:
            detection_rate_impact = 0.0

        return PatternLearningInsight(
            patterns=patterns,
            migration_candidates=[p.id for p in candidates],
            estimated_cost_reduction=round(estimated_cost_reduction, 4),
            detection_rate_impact=round(detection_rate_impact, 4),
        )

    def clear_patterns(self) -> None:
        """Clear all learned patterns."""
        pattern_count = len(self._patterns)
        self._patterns.clear()
        self._pattern_access_order.clear()
        self._feature_hashes.clear()
        logger.info(f"Cleared {pattern_count} patterns")

    def get_pattern(self, pattern_id: str) -> Optional[StageFailurePattern]:
        """Get a specific pattern by ID.

        Args:
            pattern_id: The pattern identifier.

        Returns:
            The pattern if found, None otherwise.
        """
        return self._patterns.get(pattern_id)

    def get_patterns_by_stage(self, stage: str) -> List[StageFailurePattern]:
        """Get all patterns for a specific stage.

        Args:
            stage: The stage name.

        Returns:
            List of patterns for the stage.
        """
        return [p for p in self._patterns.values() if p.stage == stage]

    def get_patterns_by_type(self, pattern_type: str) -> List[StageFailurePattern]:
        """Get all patterns of a specific type.

        Args:
            pattern_type: The pattern type.

        Returns:
            List of patterns of the specified type.
        """
        return [p for p in self._patterns.values() if p.pattern_type == pattern_type]

    def export_patterns(self) -> List[Dict[str, Any]]:
        """Export all patterns as dictionaries.

        Returns:
            List of pattern dictionaries.
        """
        return [p.to_dict() for p in self._patterns.values()]

    def import_patterns(self, patterns: List[Dict[str, Any]]) -> int:
        """Import patterns from dictionaries.

        Args:
            patterns: List of pattern dictionaries.

        Returns:
            Number of patterns imported.
        """
        imported = 0
        for pattern_dict in patterns:
            try:
                pattern = StageFailurePattern.from_dict(pattern_dict)
                self._patterns[pattern.id] = pattern

                # Update mappings
                feature_hash = self._hash_features(pattern.features)
                pattern_key = f"{pattern.stage}:{pattern.pattern_type}:{feature_hash}"
                self._feature_hashes[pattern_key] = pattern.id
                self._pattern_access_order.append(pattern.id)

                imported += 1
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to import pattern: {e}")

        self._enforce_max_patterns()
        logger.info(f"Imported {imported} patterns")
        return imported

    # Helper methods

    def _extract_numeric_features(
        self,
        data: Dict[str, Any],
        prefix: str = "",
    ) -> Dict[str, float]:
        """Recursively extract numeric features from data.

        Args:
            data: Input data dictionary.
            prefix: Current path prefix.

        Returns:
            Dictionary mapping feature paths to numeric values.
        """
        features = {}

        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                features[path] = value
            elif isinstance(value, dict):
                features.update(self._extract_numeric_features(value, path))

        return features

    def _extract_all_features(
        self,
        data: Dict[str, Any],
        prefix: str = "",
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        """Recursively extract all features from data.

        Args:
            data: Input data dictionary.
            prefix: Current path prefix.
            max_depth: Maximum recursion depth.

        Returns:
            Dictionary mapping feature paths to values.
        """
        if max_depth <= 0:
            return {}

        features = {}

        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, (str, int, float, bool)):
                features[path] = value
            elif isinstance(value, dict):
                features.update(self._extract_all_features(value, path, max_depth - 1))
            elif isinstance(value, list) and len(value) > 0:
                features[f"{path}.__len__"] = len(value)
                # Sample first element if it's a simple type
                if isinstance(value[0], (str, int, float, bool)):
                    features[f"{path}[0]"] = value[0]

        return features

    def _find_timestamp_fields(
        self,
        data: Dict[str, Any],
        prefix: str = "",
    ) -> Dict[str, Any]:
        """Find fields that appear to be timestamps.

        Args:
            data: Input data dictionary.
            prefix: Current path prefix.

        Returns:
            Dictionary mapping timestamp field paths to values.
        """
        timestamps = {}
        timestamp_keywords = {"time", "date", "timestamp", "created", "updated", "at"}

        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            key_lower = key.lower()

            # Check if key looks like a timestamp field
            if any(kw in key_lower for kw in timestamp_keywords):
                timestamps[path] = value
            elif isinstance(value, dict):
                timestamps.update(self._find_timestamp_fields(value, path))

        return timestamps

    def _hash_features(self, features: Dict[str, Any]) -> str:
        """Generate a stable hash for feature dictionary.

        Args:
            features: Feature dictionary.

        Returns:
            Hash string.
        """
        # Sort keys for consistency
        sorted_items = sorted(
            (k, str(v)) for k, v in features.items()
            if isinstance(v, (str, int, float, bool))
        )
        content = str(sorted_items).encode()
        return hashlib.md5(content).hexdigest()[:12]

    def __repr__(self) -> str:
        return (
            f"PatternExtractor(patterns={len(self._patterns)}, "
            f"types={len(set(p.pattern_type for p in self._patterns.values()))})"
        )
