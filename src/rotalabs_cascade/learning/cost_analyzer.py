"""Cost analysis for pattern migrations in cascade systems.

This module calculates ROI and cost metrics for migrating patterns between
processing stages. It provides domain-agnostic cost modeling to help determine
when patterns should be promoted to earlier (cheaper) stages or demoted to
later (more sophisticated) stages.

Example:
    >>> from rotalabs_cascade.learning.cost_analyzer import CostAnalyzer, StageCost
    >>> analyzer = CostAnalyzer()
    >>> # Customize stage costs if needed
    >>> analyzer.set_stage_cost("CUSTOM_STAGE", StageCost(
    ...     stage="CUSTOM_STAGE",
    ...     base_cost=10.0,
    ...     latency_ms=50.0,
    ...     resource_units=2.0
    ... ))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from rotalabs_cascade.learning.pattern_extractor import StageFailurePattern

logger = logging.getLogger(__name__)


@dataclass
class StageCost:
    """Cost characteristics for a processing stage.

    Attributes:
        stage: Stage identifier name.
        base_cost: Relative cost units (baseline stage = 1.0).
        latency_ms: Typical processing latency in milliseconds.
        resource_units: Compute/memory resource consumption units.
    """

    stage: str
    base_cost: float
    latency_ms: float
    resource_units: float

    def __post_init__(self) -> None:
        """Validate cost values are non-negative."""
        if self.base_cost < 0:
            raise ValueError(f"base_cost must be non-negative, got {self.base_cost}")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")
        if self.resource_units < 0:
            raise ValueError(f"resource_units must be non-negative, got {self.resource_units}")


@dataclass
class MigrationROI:
    """ROI calculation for migrating a pattern to a different stage.

    Attributes:
        pattern_id: Unique identifier for the pattern being analyzed.
        source_stage: Current stage where pattern is being handled.
        target_stage: Proposed stage to migrate pattern to.
        current_cost_per_item: Cost per item at current stage.
        projected_cost_per_item: Expected cost per item at target stage.
        estimated_volume: Expected items per period to process.
        cost_reduction_absolute: Total cost savings per period.
        cost_reduction_percentage: Percentage reduction in costs.
        detection_rate_change: Change in detection rate (positive = improvement).
        false_positive_risk: Risk of false positives at target stage (0.0-1.0).
        payback_items: Number of items needed to break even on migration effort.
        recommendation: Migration recommendation (MIGRATE, MONITOR, REJECT).
    """

    pattern_id: str
    source_stage: str
    target_stage: str
    current_cost_per_item: float
    projected_cost_per_item: float
    estimated_volume: int
    cost_reduction_absolute: float
    cost_reduction_percentage: float
    detection_rate_change: float
    false_positive_risk: float
    payback_items: int
    recommendation: str

    def __post_init__(self) -> None:
        """Validate recommendation value."""
        valid_recommendations = {"MIGRATE", "MONITOR", "REJECT"}
        if self.recommendation not in valid_recommendations:
            raise ValueError(
                f"recommendation must be one of {valid_recommendations}, "
                f"got {self.recommendation}"
            )


# Default stage cost configuration
# These represent relative costs where RULES is the baseline (1x)
DEFAULT_STAGE_COSTS: Dict[str, StageCost] = {
    "RULES": StageCost(
        stage="RULES",
        base_cost=1.0,
        latency_ms=1.0,
        resource_units=0.1,
    ),
    "STATISTICAL_ML": StageCost(
        stage="STATISTICAL_ML",
        base_cost=5.0,
        latency_ms=10.0,
        resource_units=0.5,
    ),
    "SINGLE_AI": StageCost(
        stage="SINGLE_AI",
        base_cost=25.0,
        latency_ms=100.0,
        resource_units=2.0,
    ),
    "POD": StageCost(
        stage="POD",
        base_cost=100.0,
        latency_ms=500.0,
        resource_units=10.0,
    ),
    "ADVERSARIAL": StageCost(
        stage="ADVERSARIAL",
        base_cost=500.0,
        latency_ms=2000.0,
        resource_units=50.0,
    ),
}


class CostAnalyzer:
    """Analyzes migration costs and ROI for cascade pattern optimization.

    The CostAnalyzer evaluates whether patterns should be migrated between
    processing stages based on cost efficiency, detection rates, and risk
    factors. It uses configurable stage costs to model the tradeoffs between
    earlier (cheaper but less sophisticated) and later (expensive but more
    capable) stages.

    Example:
        >>> analyzer = CostAnalyzer()
        >>> # Calculate ROI for a single pattern
        >>> roi = analyzer.calculate_migration_roi(pattern, "RULES", volume=10000)
        >>> print(f"Recommendation: {roi.recommendation}")
        >>> print(f"Savings: {roi.cost_reduction_absolute:.2f} per period")

    Attributes:
        stage_costs: Dictionary mapping stage names to their cost configurations.
    """

    # Thresholds for migration recommendations
    MIGRATE_THRESHOLD_PERCENTAGE: float = 20.0  # Minimum % savings to recommend MIGRATE
    MONITOR_THRESHOLD_PERCENTAGE: float = 5.0   # Minimum % savings to recommend MONITOR
    MAX_FALSE_POSITIVE_RISK: float = 0.1        # Maximum acceptable false positive risk
    DEFAULT_MIGRATION_EFFORT: float = 100.0     # Default cost units for migration effort

    def __init__(
        self,
        stage_costs: Optional[Dict[str, StageCost]] = None,
    ) -> None:
        """Initialize the cost analyzer.

        Args:
            stage_costs: Optional custom stage cost configuration.
                If not provided, uses DEFAULT_STAGE_COSTS.
        """
        if stage_costs is not None:
            self.stage_costs = dict(stage_costs)
        else:
            self.stage_costs = dict(DEFAULT_STAGE_COSTS)
        logger.debug(
            "Initialized CostAnalyzer with %d stage configurations",
            len(self.stage_costs),
        )

    def set_stage_cost(self, stage: str, cost: StageCost) -> None:
        """Set or update the cost configuration for a stage.

        Args:
            stage: Stage identifier name.
            cost: Cost configuration for the stage.
        """
        if cost.stage != stage:
            raise ValueError(
                f"StageCost.stage ({cost.stage}) must match stage parameter ({stage})"
            )
        self.stage_costs[stage] = cost
        logger.debug("Updated cost configuration for stage: %s", stage)

    def calculate_migration_roi(
        self,
        pattern: "StageFailurePattern",
        target_stage: str,
        volume: int,
        migration_effort: Optional[float] = None,
    ) -> MigrationROI:
        """Calculate ROI for migrating a pattern to a different stage.

        Args:
            pattern: The stage failure pattern to analyze.
            target_stage: Stage to potentially migrate the pattern to.
            volume: Estimated number of items per period.
            migration_effort: Cost units for migration effort (default: 100.0).

        Returns:
            MigrationROI with cost analysis and recommendation.

        Raises:
            ValueError: If source or target stage is not configured.
        """
        source_stage = pattern.stage
        if source_stage not in self.stage_costs:
            raise ValueError(f"Source stage '{source_stage}' not in stage_costs configuration")
        if target_stage not in self.stage_costs:
            raise ValueError(f"Target stage '{target_stage}' not in stage_costs configuration")

        source_cost = self.stage_costs[source_stage]
        target_cost = self.stage_costs[target_stage]

        # Calculate per-item costs
        current_cost_per_item = source_cost.base_cost
        projected_cost_per_item = target_cost.base_cost

        # Calculate cost reduction
        cost_reduction_per_item = current_cost_per_item - projected_cost_per_item
        cost_reduction_absolute = cost_reduction_per_item * volume

        if current_cost_per_item > 0:
            cost_reduction_percentage = (cost_reduction_per_item / current_cost_per_item) * 100
        else:
            cost_reduction_percentage = 0.0

        # Estimate detection rate change based on pattern confidence and stage capability
        # Moving to an earlier (cheaper) stage typically means lower detection capability
        # Moving to a later (more expensive) stage typically means higher detection capability
        stage_order = list(self.stage_costs.keys())
        source_idx = stage_order.index(source_stage) if source_stage in stage_order else 0
        target_idx = stage_order.index(target_stage) if target_stage in stage_order else 0

        # Detection rate change: negative if moving to earlier stage, positive if moving to later
        # Scaled by pattern confidence (high confidence patterns are safer to move)
        confidence = getattr(pattern, "confidence", 0.5)
        detection_rate_change = (target_idx - source_idx) * 0.05 * (1 - confidence)

        # False positive risk increases when moving to earlier (less sophisticated) stages
        # Risk is inversely proportional to pattern confidence
        if target_idx < source_idx:
            # Moving to earlier stage - higher false positive risk
            false_positive_risk = min(1.0, (1 - confidence) * 0.2 * (source_idx - target_idx))
        else:
            # Moving to later stage or same stage - minimal false positive risk
            false_positive_risk = 0.01

        # Calculate payback items (items needed to recover migration effort)
        effort = migration_effort if migration_effort is not None else self.DEFAULT_MIGRATION_EFFORT
        if cost_reduction_per_item > 0:
            payback_items = int(effort / cost_reduction_per_item) + 1
        else:
            # No cost reduction or cost increase - infinite payback
            payback_items = volume * 1000  # Effectively "never pays back"

        # Determine recommendation
        recommendation = self._determine_recommendation(
            cost_reduction_percentage=cost_reduction_percentage,
            false_positive_risk=false_positive_risk,
            confidence=confidence,
            volume=volume,
            payback_items=payback_items,
        )

        roi = MigrationROI(
            pattern_id=pattern.id,
            source_stage=source_stage,
            target_stage=target_stage,
            current_cost_per_item=current_cost_per_item,
            projected_cost_per_item=projected_cost_per_item,
            estimated_volume=volume,
            cost_reduction_absolute=cost_reduction_absolute,
            cost_reduction_percentage=cost_reduction_percentage,
            detection_rate_change=detection_rate_change,
            false_positive_risk=false_positive_risk,
            payback_items=payback_items,
            recommendation=recommendation,
        )

        logger.info(
            "Migration ROI calculated for pattern %s: %s -> %s, recommendation=%s, "
            "savings=%.2f (%.1f%%)",
            pattern.id,
            source_stage,
            target_stage,
            recommendation,
            cost_reduction_absolute,
            cost_reduction_percentage,
        )

        return roi

    def _determine_recommendation(
        self,
        cost_reduction_percentage: float,
        false_positive_risk: float,
        confidence: float,
        volume: int,
        payback_items: int,
    ) -> str:
        """Determine migration recommendation based on metrics.

        Args:
            cost_reduction_percentage: Percentage cost savings.
            false_positive_risk: Risk of false positives (0.0-1.0).
            confidence: Pattern confidence score (0.0-1.0).
            volume: Estimated items per period.
            payback_items: Items needed to break even.

        Returns:
            Recommendation string: MIGRATE, MONITOR, or REJECT.
        """
        # Reject if false positive risk is too high
        if false_positive_risk > self.MAX_FALSE_POSITIVE_RISK:
            logger.debug(
                "Rejecting migration: false_positive_risk (%.2f) > threshold (%.2f)",
                false_positive_risk,
                self.MAX_FALSE_POSITIVE_RISK,
            )
            return "REJECT"

        # Reject if payback period is unreasonable (more than 10x the volume)
        if payback_items > volume * 10:
            logger.debug(
                "Rejecting migration: payback_items (%d) > 10x volume (%d)",
                payback_items,
                volume * 10,
            )
            return "REJECT"

        # Reject if cost reduction is negative or negligible
        if cost_reduction_percentage < self.MONITOR_THRESHOLD_PERCENTAGE:
            logger.debug(
                "Rejecting migration: cost_reduction (%.1f%%) < threshold (%.1f%%)",
                cost_reduction_percentage,
                self.MONITOR_THRESHOLD_PERCENTAGE,
            )
            return "REJECT"

        # Recommend migration if savings are significant and confidence is high
        if (
            cost_reduction_percentage >= self.MIGRATE_THRESHOLD_PERCENTAGE
            and confidence >= 0.8
        ):
            logger.debug(
                "Recommending migration: cost_reduction=%.1f%%, confidence=%.2f",
                cost_reduction_percentage,
                confidence,
            )
            return "MIGRATE"

        # Monitor for patterns with moderate savings or confidence
        logger.debug(
            "Recommending monitoring: cost_reduction=%.1f%%, confidence=%.2f",
            cost_reduction_percentage,
            confidence,
        )
        return "MONITOR"

    def analyze_all_candidates(
        self,
        candidates: List["StageFailurePattern"],
        target_stage: str = "RULES",
        volume: int = 1000,
    ) -> List[MigrationROI]:
        """Analyze all candidate patterns for migration to a target stage.

        Args:
            candidates: List of stage failure patterns to analyze.
            target_stage: Target stage for migration (default: RULES).
            volume: Estimated items per period for each pattern.

        Returns:
            List of MigrationROI objects for each candidate.
        """
        results = []
        for pattern in candidates:
            try:
                roi = self.calculate_migration_roi(
                    pattern=pattern,
                    target_stage=target_stage,
                    volume=volume,
                )
                results.append(roi)
            except ValueError as e:
                logger.warning(
                    "Skipping pattern %s: %s",
                    getattr(pattern, "pattern_id", "unknown"),
                    str(e),
                )

        logger.info(
            "Analyzed %d candidates for migration to %s",
            len(results),
            target_stage,
        )
        return results

    def get_total_potential_savings(
        self,
        roi_list: List[MigrationROI],
    ) -> Dict[str, Any]:
        """Calculate total potential savings from a list of ROI analyses.

        Args:
            roi_list: List of MigrationROI objects to aggregate.

        Returns:
            Dictionary containing:
                - total_absolute_savings: Sum of all cost reductions.
                - total_volume: Sum of all estimated volumes.
                - average_reduction_percentage: Weighted average cost reduction.
                - migrate_count: Number of MIGRATE recommendations.
                - monitor_count: Number of MONITOR recommendations.
                - reject_count: Number of REJECT recommendations.
                - patterns_by_recommendation: Pattern IDs grouped by recommendation.
        """
        if not roi_list:
            return {
                "total_absolute_savings": 0.0,
                "total_volume": 0,
                "average_reduction_percentage": 0.0,
                "migrate_count": 0,
                "monitor_count": 0,
                "reject_count": 0,
                "patterns_by_recommendation": {
                    "MIGRATE": [],
                    "MONITOR": [],
                    "REJECT": [],
                },
            }

        total_absolute_savings = sum(roi.cost_reduction_absolute for roi in roi_list)
        total_volume = sum(roi.estimated_volume for roi in roi_list)
        total_current_cost = sum(
            roi.current_cost_per_item * roi.estimated_volume for roi in roi_list
        )

        if total_current_cost > 0:
            average_reduction_percentage = (total_absolute_savings / total_current_cost) * 100
        else:
            average_reduction_percentage = 0.0

        patterns_by_recommendation: Dict[str, List[str]] = {
            "MIGRATE": [],
            "MONITOR": [],
            "REJECT": [],
        }

        for roi in roi_list:
            patterns_by_recommendation[roi.recommendation].append(roi.pattern_id)

        migrate_count = len(patterns_by_recommendation["MIGRATE"])
        monitor_count = len(patterns_by_recommendation["MONITOR"])
        reject_count = len(patterns_by_recommendation["REJECT"])

        result = {
            "total_absolute_savings": total_absolute_savings,
            "total_volume": total_volume,
            "average_reduction_percentage": average_reduction_percentage,
            "migrate_count": migrate_count,
            "monitor_count": monitor_count,
            "reject_count": reject_count,
            "patterns_by_recommendation": patterns_by_recommendation,
        }

        logger.info(
            "Total potential savings: %.2f (%.1f%% avg reduction) across %d patterns "
            "(MIGRATE: %d, MONITOR: %d, REJECT: %d)",
            total_absolute_savings,
            average_reduction_percentage,
            len(roi_list),
            migrate_count,
            monitor_count,
            reject_count,
        )

        return result

    def rank_by_roi(
        self,
        roi_list: List[MigrationROI],
        ascending: bool = False,
    ) -> List[MigrationROI]:
        """Rank migration candidates by ROI.

        Patterns are ranked primarily by cost reduction absolute value,
        with recommendation and false positive risk as secondary factors.

        Args:
            roi_list: List of MigrationROI objects to rank.
            ascending: If True, rank from lowest to highest ROI.

        Returns:
            Sorted list of MigrationROI objects.
        """
        # Recommendation priority: MIGRATE > MONITOR > REJECT
        recommendation_priority = {"MIGRATE": 0, "MONITOR": 1, "REJECT": 2}

        def sort_key(roi: MigrationROI) -> tuple:
            return (
                recommendation_priority.get(roi.recommendation, 3),
                -roi.cost_reduction_absolute if not ascending else roi.cost_reduction_absolute,
                roi.false_positive_risk,
                -roi.cost_reduction_percentage if not ascending else roi.cost_reduction_percentage,
            )

        sorted_list = sorted(roi_list, key=sort_key)

        logger.debug(
            "Ranked %d patterns by ROI (ascending=%s)",
            len(sorted_list),
            ascending,
        )

        return sorted_list
