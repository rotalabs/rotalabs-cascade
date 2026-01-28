"""Learning module for cascade rule generation and proposal workflow.

This subpackage provides pattern extraction, rule generation, cost analysis,
and proposal management for optimizing cascade routing decisions based on
historical execution data.
"""

from rotalabs_cascade.learning.cost_analyzer import (
    CostAnalyzer,
    MigrationROI,
    StageCost,
)
from rotalabs_cascade.learning.pattern_extractor import (
    PatternConfig,
    PatternExtractor,
    PatternLearningInsight,
    PatternType,
    StageFailurePattern,
)
from rotalabs_cascade.learning.proposal import (
    ProposalManager,
    ProposalStatus,
    RuleProposal,
)
from rotalabs_cascade.learning.rule_generator import (
    GeneratedRule,
    RuleGenerator,
    RuleTemplate,
)

__all__ = [
    # Pattern extraction
    "PatternConfig",
    "PatternExtractor",
    "PatternLearningInsight",
    "PatternType",
    "StageFailurePattern",
    # Rule generation
    "GeneratedRule",
    "RuleGenerator",
    "RuleTemplate",
    # Cost analysis
    "CostAnalyzer",
    "MigrationROI",
    "StageCost",
    # Proposals
    "ProposalManager",
    "ProposalStatus",
    "RuleProposal",
]
