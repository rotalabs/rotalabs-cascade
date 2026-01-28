"""Rule proposal and human approval workflow for cascade learning system.

This module handles the lifecycle of machine-generated routing rules, from initial
proposal through human review, A/B testing, activation, and eventual deprecation.

The workflow supports:
- Creating proposals from generated rules with ROI analysis
- Human review with approve/reject decisions
- A/B testing before full activation
- Recording test results and metrics
- Activating approved rules
- Deprecating obsolete rules

Example:
    >>> from rotalabs_cascade.learning.proposal import ProposalManager
    >>> manager = ProposalManager()
    >>> proposal = manager.create_proposal(generated_rule, roi_analysis)
    >>> manager.approve(proposal.proposal_id, reviewer="reviewer@example.com")
    >>> manager.start_testing(proposal.proposal_id)
    >>> manager.record_test_results(proposal.proposal_id, {"accuracy": 0.95})
    >>> manager.activate(proposal.proposal_id)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rotalabs_cascade.learning.rule_generator import GeneratedRule
    from rotalabs_cascade.learning.cost_analyzer import MigrationROI

logger = logging.getLogger(__name__)


class ProposalStatus(str, Enum):
    """Status of a rule proposal in the approval workflow.

    Attributes:
        PENDING_REVIEW: Proposal awaiting human review.
        APPROVED: Proposal approved by reviewer, ready for testing or activation.
        REJECTED: Proposal rejected by reviewer.
        TESTING: Proposal in A/B testing phase.
        ACTIVE: Proposal deployed and active in production.
        DEPRECATED: Proposal deprecated and no longer in use.
    """

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    TESTING = "testing"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass
class RuleProposal:
    """A proposal for a machine-generated routing rule.

    This dataclass captures all metadata about a rule proposal throughout its
    lifecycle, from initial creation through review, testing, and activation.

    Attributes:
        proposal_id: Unique identifier for this proposal.
        generated_rule: The machine-generated rule being proposed.
        roi_analysis: ROI analysis supporting this proposal.
        status: Current status in the approval workflow.
        created_at: Timestamp when proposal was created.
        reviewed_at: Timestamp when proposal was reviewed (approved/rejected).
        reviewer: Identifier of the person who reviewed the proposal.
        review_notes: Notes from the reviewer explaining their decision.
        test_results: Results from A/B testing phase.
        activated_at: Timestamp when proposal was activated in production.
    """

    proposal_id: str
    generated_rule: Any  # GeneratedRule - using Any for forward compatibility
    roi_analysis: Any  # MigrationROI - using Any for forward compatibility
    status: ProposalStatus = ProposalStatus.PENDING_REVIEW
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reviewed_at: Optional[datetime] = None
    reviewer: Optional[str] = None
    review_notes: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    activated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert proposal to dictionary for serialization.

        Returns:
            Dictionary representation of the proposal.
        """
        return {
            "proposal_id": self.proposal_id,
            "generated_rule": (
                self.generated_rule.to_dict()
                if hasattr(self.generated_rule, "to_dict")
                else self.generated_rule
            ),
            "roi_analysis": (
                self.roi_analysis.to_dict()
                if hasattr(self.roi_analysis, "to_dict")
                else self.roi_analysis
            ),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewer": self.reviewer,
            "review_notes": self.review_notes,
            "test_results": self.test_results,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleProposal":
        """Create proposal from dictionary.

        Args:
            data: Dictionary containing proposal data.

        Returns:
            RuleProposal instance.
        """
        return cls(
            proposal_id=data["proposal_id"],
            generated_rule=data["generated_rule"],
            roi_analysis=data["roi_analysis"],
            status=ProposalStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            reviewed_at=(
                datetime.fromisoformat(data["reviewed_at"])
                if data.get("reviewed_at")
                else None
            ),
            reviewer=data.get("reviewer"),
            review_notes=data.get("review_notes"),
            test_results=data.get("test_results"),
            activated_at=(
                datetime.fromisoformat(data["activated_at"])
                if data.get("activated_at")
                else None
            ),
        )


class ProposalManager:
    """Manages the lifecycle of rule proposals through the approval workflow.

    This class provides methods for creating, reviewing, testing, and activating
    rule proposals. It supports optional file-based persistence for proposals.

    Attributes:
        storage_path: Optional path for persisting proposals to disk.
        _proposals: In-memory dictionary of proposals keyed by proposal_id.

    Example:
        >>> manager = ProposalManager(storage_path=Path("./proposals"))
        >>> proposal = manager.create_proposal(rule, roi)
        >>> pending = manager.get_pending_proposals()
        >>> manager.approve(proposal.proposal_id, "reviewer@example.com")
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        """Initialize the proposal manager.

        Args:
            storage_path: Optional path for file-based persistence. If provided,
                proposals will be saved to and loaded from this directory.
        """
        self.storage_path = storage_path
        self._proposals: Dict[str, RuleProposal] = {}

        if self.storage_path:
            self.storage_path = Path(self.storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_from_storage()

        logger.info(
            "ProposalManager initialized",
            extra={
                "storage_path": str(self.storage_path) if self.storage_path else None,
                "proposal_count": len(self._proposals),
            },
        )

    def _load_from_storage(self) -> None:
        """Load proposals from storage directory."""
        if not self.storage_path or not self.storage_path.exists():
            return

        proposals_file = self.storage_path / "proposals.json"
        if proposals_file.exists():
            try:
                with open(proposals_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for proposal_data in data.get("proposals", []):
                        proposal = RuleProposal.from_dict(proposal_data)
                        self._proposals[proposal.proposal_id] = proposal
                logger.debug(
                    "Loaded proposals from storage",
                    extra={"count": len(self._proposals)},
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(
                    "Failed to load proposals from storage",
                    extra={"error": str(e)},
                )

    def _save_to_storage(self) -> None:
        """Save proposals to storage directory."""
        if not self.storage_path:
            return

        proposals_file = self.storage_path / "proposals.json"
        try:
            data = {
                "proposals": [p.to_dict() for p in self._proposals.values()],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(proposals_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(
                "Saved proposals to storage",
                extra={"count": len(self._proposals)},
            )
        except (IOError, TypeError) as e:
            logger.error(
                "Failed to save proposals to storage",
                extra={"error": str(e)},
            )

    def create_proposal(
        self,
        rule: Any,  # GeneratedRule
        roi: Any,  # MigrationROI
    ) -> RuleProposal:
        """Create a new rule proposal for review.

        Args:
            rule: The machine-generated rule to propose.
            roi: ROI analysis supporting this proposal.

        Returns:
            The created RuleProposal with PENDING_REVIEW status.
        """
        proposal_id = str(uuid.uuid4())
        proposal = RuleProposal(
            proposal_id=proposal_id,
            generated_rule=rule,
            roi_analysis=roi,
            status=ProposalStatus.PENDING_REVIEW,
            created_at=datetime.now(timezone.utc),
        )

        self._proposals[proposal_id] = proposal
        self._save_to_storage()

        logger.info(
            "Created new proposal",
            extra={"proposal_id": proposal_id},
        )

        return proposal

    def get_pending_proposals(self) -> List[RuleProposal]:
        """Get all proposals pending review.

        Returns:
            List of proposals with PENDING_REVIEW status, sorted by creation time.
        """
        pending = [
            p for p in self._proposals.values()
            if p.status == ProposalStatus.PENDING_REVIEW
        ]
        return sorted(pending, key=lambda p: p.created_at)

    def approve(
        self,
        proposal_id: str,
        reviewer: str,
        notes: Optional[str] = None,
    ) -> RuleProposal:
        """Approve a pending proposal.

        Args:
            proposal_id: ID of the proposal to approve.
            reviewer: Identifier of the person approving.
            notes: Optional notes explaining the approval decision.

        Returns:
            The updated RuleProposal with APPROVED status.

        Raises:
            KeyError: If proposal_id not found.
            ValueError: If proposal is not in PENDING_REVIEW status.
        """
        proposal = self._get_proposal_or_raise(proposal_id)

        if proposal.status != ProposalStatus.PENDING_REVIEW:
            raise ValueError(
                f"Cannot approve proposal in {proposal.status.value} status. "
                "Only PENDING_REVIEW proposals can be approved."
            )

        proposal.status = ProposalStatus.APPROVED
        proposal.reviewed_at = datetime.now(timezone.utc)
        proposal.reviewer = reviewer
        proposal.review_notes = notes

        self._save_to_storage()

        logger.info(
            "Proposal approved",
            extra={
                "proposal_id": proposal_id,
                "reviewer": reviewer,
            },
        )

        return proposal

    def reject(
        self,
        proposal_id: str,
        reviewer: str,
        notes: str,
    ) -> RuleProposal:
        """Reject a pending proposal.

        Args:
            proposal_id: ID of the proposal to reject.
            reviewer: Identifier of the person rejecting.
            notes: Required notes explaining the rejection reason.

        Returns:
            The updated RuleProposal with REJECTED status.

        Raises:
            KeyError: If proposal_id not found.
            ValueError: If proposal is not in PENDING_REVIEW status.
        """
        proposal = self._get_proposal_or_raise(proposal_id)

        if proposal.status != ProposalStatus.PENDING_REVIEW:
            raise ValueError(
                f"Cannot reject proposal in {proposal.status.value} status. "
                "Only PENDING_REVIEW proposals can be rejected."
            )

        proposal.status = ProposalStatus.REJECTED
        proposal.reviewed_at = datetime.now(timezone.utc)
        proposal.reviewer = reviewer
        proposal.review_notes = notes

        self._save_to_storage()

        logger.info(
            "Proposal rejected",
            extra={
                "proposal_id": proposal_id,
                "reviewer": reviewer,
                "reason": notes,
            },
        )

        return proposal

    def start_testing(self, proposal_id: str) -> RuleProposal:
        """Start A/B testing for an approved proposal.

        Args:
            proposal_id: ID of the proposal to begin testing.

        Returns:
            The updated RuleProposal with TESTING status.

        Raises:
            KeyError: If proposal_id not found.
            ValueError: If proposal is not in APPROVED status.
        """
        proposal = self._get_proposal_or_raise(proposal_id)

        if proposal.status != ProposalStatus.APPROVED:
            raise ValueError(
                f"Cannot start testing for proposal in {proposal.status.value} status. "
                "Only APPROVED proposals can be tested."
            )

        proposal.status = ProposalStatus.TESTING
        proposal.test_results = {}

        self._save_to_storage()

        logger.info(
            "Started testing for proposal",
            extra={"proposal_id": proposal_id},
        )

        return proposal

    def record_test_results(
        self,
        proposal_id: str,
        results: Dict[str, Any],
    ) -> RuleProposal:
        """Record test results for a proposal in testing.

        Args:
            proposal_id: ID of the proposal being tested.
            results: Dictionary of test results and metrics.

        Returns:
            The updated RuleProposal with recorded test results.

        Raises:
            KeyError: If proposal_id not found.
            ValueError: If proposal is not in TESTING status.
        """
        proposal = self._get_proposal_or_raise(proposal_id)

        if proposal.status != ProposalStatus.TESTING:
            raise ValueError(
                f"Cannot record test results for proposal in {proposal.status.value} "
                "status. Only TESTING proposals can have results recorded."
            )

        if proposal.test_results is None:
            proposal.test_results = {}

        proposal.test_results.update(results)

        self._save_to_storage()

        logger.info(
            "Recorded test results for proposal",
            extra={
                "proposal_id": proposal_id,
                "result_keys": list(results.keys()),
            },
        )

        return proposal

    def activate(self, proposal_id: str) -> RuleProposal:
        """Activate a tested proposal for production use.

        Args:
            proposal_id: ID of the proposal to activate.

        Returns:
            The updated RuleProposal with ACTIVE status.

        Raises:
            KeyError: If proposal_id not found.
            ValueError: If proposal is not in TESTING or APPROVED status.
        """
        proposal = self._get_proposal_or_raise(proposal_id)

        if proposal.status not in (ProposalStatus.TESTING, ProposalStatus.APPROVED):
            raise ValueError(
                f"Cannot activate proposal in {proposal.status.value} status. "
                "Only TESTING or APPROVED proposals can be activated."
            )

        proposal.status = ProposalStatus.ACTIVE
        proposal.activated_at = datetime.now(timezone.utc)

        self._save_to_storage()

        logger.info(
            "Activated proposal",
            extra={"proposal_id": proposal_id},
        )

        return proposal

    def deprecate(self, proposal_id: str, reason: str) -> RuleProposal:
        """Deprecate an active proposal.

        Args:
            proposal_id: ID of the proposal to deprecate.
            reason: Reason for deprecation.

        Returns:
            The updated RuleProposal with DEPRECATED status.

        Raises:
            KeyError: If proposal_id not found.
            ValueError: If proposal is not in ACTIVE status.
        """
        proposal = self._get_proposal_or_raise(proposal_id)

        if proposal.status != ProposalStatus.ACTIVE:
            raise ValueError(
                f"Cannot deprecate proposal in {proposal.status.value} status. "
                "Only ACTIVE proposals can be deprecated."
            )

        proposal.status = ProposalStatus.DEPRECATED
        if proposal.review_notes:
            proposal.review_notes += f"\n\nDeprecation reason: {reason}"
        else:
            proposal.review_notes = f"Deprecation reason: {reason}"

        self._save_to_storage()

        logger.info(
            "Deprecated proposal",
            extra={
                "proposal_id": proposal_id,
                "reason": reason,
            },
        )

        return proposal

    def get_proposal(self, proposal_id: str) -> Optional[RuleProposal]:
        """Get a proposal by ID.

        Args:
            proposal_id: ID of the proposal to retrieve.

        Returns:
            The RuleProposal if found, None otherwise.
        """
        return self._proposals.get(proposal_id)

    def get_active_rules(self) -> List[RuleProposal]:
        """Get all active rule proposals.

        Returns:
            List of proposals with ACTIVE status, sorted by activation time.
        """
        active = [
            p for p in self._proposals.values()
            if p.status == ProposalStatus.ACTIVE
        ]
        return sorted(
            active,
            key=lambda p: p.activated_at or p.created_at,
        )

    def export_proposals(self, path: Path) -> None:
        """Export all proposals to a JSON file.

        Args:
            path: Path to write the JSON file.
        """
        path = Path(path)
        data = {
            "proposals": [p.to_dict() for p in self._proposals.values()],
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_count": len(self._proposals),
            "status_counts": self._get_status_counts(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "Exported proposals",
            extra={
                "path": str(path),
                "count": len(self._proposals),
            },
        )

    def import_proposals(self, path: Path) -> None:
        """Import proposals from a JSON file.

        Args:
            path: Path to the JSON file to import.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        imported_count = 0
        for proposal_data in data.get("proposals", []):
            try:
                proposal = RuleProposal.from_dict(proposal_data)
                self._proposals[proposal.proposal_id] = proposal
                imported_count += 1
            except (KeyError, ValueError) as e:
                logger.warning(
                    "Failed to import proposal",
                    extra={
                        "error": str(e),
                        "proposal_id": proposal_data.get("proposal_id", "unknown"),
                    },
                )

        self._save_to_storage()

        logger.info(
            "Imported proposals",
            extra={
                "path": str(path),
                "imported_count": imported_count,
            },
        )

    def _get_proposal_or_raise(self, proposal_id: str) -> RuleProposal:
        """Get a proposal by ID or raise KeyError.

        Args:
            proposal_id: ID of the proposal to retrieve.

        Returns:
            The RuleProposal.

        Raises:
            KeyError: If proposal_id not found.
        """
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Proposal not found: {proposal_id}")
        return proposal

    def _get_status_counts(self) -> Dict[str, int]:
        """Get counts of proposals by status.

        Returns:
            Dictionary mapping status values to counts.
        """
        counts: Dict[str, int] = {}
        for proposal in self._proposals.values():
            status_value = proposal.status.value
            counts[status_value] = counts.get(status_value, 0) + 1
        return counts
