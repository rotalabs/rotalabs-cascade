"""Universal Event + Context models for domain-agnostic cascade processing.

This module provides structured models that work across all domains:
- Finance (transactions, transfers)
- Healthcare (claims, prescriptions)
- Supply Chain (orders, invoices)
- Cybersecurity (access attempts, logins)
- Content Moderation (posts, comments)
- Customer Support (tickets, requests)
- And any generic domain

The Event + Context pattern separates:
- Event: What happened (the action being evaluated)
- Context: The circumstances around it (who, where, when, history)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class DomainType(Enum):
    """Supported domain types for event processing."""

    FINANCE = "FINANCE"
    HEALTHCARE = "HEALTHCARE"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    CYBERSECURITY = "CYBERSECURITY"
    CONTENT_MODERATION = "CONTENT_MODERATION"
    CUSTOMER_SUPPORT = "CUSTOMER_SUPPORT"
    INSURANCE = "INSURANCE"
    RETAIL = "RETAIL"
    HR_RECRUITING = "HR_RECRUITING"
    GENERIC = "GENERIC"


class EventType(Enum):
    """Generic event types across domains."""

    # Finance
    TRANSACTION = "transaction"
    TRANSFER = "transfer"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    PAYMENT = "payment"

    # Healthcare
    CLAIM = "claim"
    PRESCRIPTION = "prescription"
    APPOINTMENT = "appointment"
    BILLING = "billing"

    # Supply Chain
    PURCHASE_ORDER = "purchase_order"
    INVOICE = "invoice"
    DELIVERY = "delivery"
    SHIPMENT = "shipment"

    # Cybersecurity
    ACCESS_ATTEMPT = "access_attempt"
    DATA_TRANSFER = "data_transfer"
    LOGIN = "login"
    API_CALL = "api_call"
    PERMISSION_CHANGE = "permission_change"

    # Content Moderation
    POST = "post"
    COMMENT = "comment"
    MESSAGE = "message"
    UPLOAD = "upload"
    REPORT = "report"

    # Customer Support
    TICKET = "ticket"
    INQUIRY = "inquiry"
    COMPLAINT = "complaint"
    FEEDBACK = "feedback"

    # HR/Recruiting
    APPLICATION = "application"
    RESUME = "resume"
    INTERVIEW = "interview"

    # Generic
    GENERIC_EVENT = "generic_event"
    REQUEST = "request"
    ACTION = "action"


@dataclass
class SessionContext:
    """Session-related contextual information."""

    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    login_timestamp: Optional[datetime] = None
    session_duration_minutes: Optional[int] = None
    previous_session_count: Optional[int] = None
    is_authenticated: Optional[bool] = None
    auth_method: Optional[str] = None  # password, oauth, sso, mfa


@dataclass
class DeviceContext:
    """Device-related contextual information."""

    device_id: Optional[str] = None
    device_type: Optional[str] = None  # mobile, desktop, tablet, iot
    device_info: Optional[str] = None
    operating_system: Optional[str] = None
    browser: Optional[str] = None
    app_version: Optional[str] = None
    is_trusted_device: Optional[bool] = None
    device_fingerprint: Optional[str] = None


@dataclass
class LocationContext:
    """Location-related contextual information."""

    current_location: Optional[str] = None
    registered_location: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None  # {lat, lng}
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    timezone: Optional[str] = None
    vpn_detected: Optional[bool] = None
    proxy_detected: Optional[bool] = None
    location_accuracy: Optional[str] = None  # precise, city, region, country


@dataclass
class HistoricalContext:
    """Historical behavioral contextual information."""

    account_age_days: Optional[int] = None
    previous_events_count: Optional[int] = None
    average_event_value: Optional[float] = None
    max_event_value: Optional[float] = None
    typical_event_times: Optional[List[str]] = None  # ["09:00-17:00"]
    typical_locations: Optional[List[str]] = None
    last_event_timestamp: Optional[datetime] = None
    event_frequency_per_day: Optional[float] = None
    trust_score: Optional[float] = None  # 0-1 historical trust
    risk_flags: Optional[List[str]] = None


@dataclass
class EntityContext:
    """Information about the entities involved."""

    primary_entity_id: Optional[str] = None
    primary_entity_type: Optional[str] = None  # user, customer, patient, employee
    primary_entity_verified: Optional[bool] = None
    secondary_entity_id: Optional[str] = None
    secondary_entity_type: Optional[str] = None  # merchant, provider, vendor
    secondary_entity_trust: Optional[float] = None
    relationship_age_days: Optional[int] = None


@dataclass
class EventContext:
    """Complete contextual information surrounding an event.

    This structured context enables domain-agnostic routing rules
    that work across finance, healthcare, security, content, etc.
    """

    session: SessionContext = field(default_factory=SessionContext)
    device: DeviceContext = field(default_factory=DeviceContext)
    location: LocationContext = field(default_factory=LocationContext)
    historical: HistoricalContext = field(default_factory=HistoricalContext)
    entity: EntityContext = field(default_factory=EntityContext)

    # Domain-specific context (flexible extension point)
    domain_context: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    context_timestamp: datetime = field(default_factory=datetime.now)
    context_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "session": {
                "ip_address": self.session.ip_address,
                "user_agent": self.session.user_agent,
                "session_id": self.session.session_id,
                "login_timestamp": self.session.login_timestamp.isoformat() if self.session.login_timestamp else None,
                "session_duration_minutes": self.session.session_duration_minutes,
                "previous_session_count": self.session.previous_session_count,
                "is_authenticated": self.session.is_authenticated,
                "auth_method": self.session.auth_method,
            },
            "device": {
                "device_id": self.device.device_id,
                "device_type": self.device.device_type,
                "device_info": self.device.device_info,
                "operating_system": self.device.operating_system,
                "browser": self.device.browser,
                "app_version": self.device.app_version,
                "is_trusted_device": self.device.is_trusted_device,
            },
            "location": {
                "current_location": self.location.current_location,
                "registered_location": self.location.registered_location,
                "coordinates": self.location.coordinates,
                "country": self.location.country,
                "region": self.location.region,
                "city": self.location.city,
                "timezone": self.location.timezone,
                "vpn_detected": self.location.vpn_detected,
                "proxy_detected": self.location.proxy_detected,
            },
            "historical": {
                "account_age_days": self.historical.account_age_days,
                "previous_events_count": self.historical.previous_events_count,
                "average_event_value": self.historical.average_event_value,
                "typical_event_times": self.historical.typical_event_times,
                "typical_locations": self.historical.typical_locations,
                "last_event_timestamp": self.historical.last_event_timestamp.isoformat() if self.historical.last_event_timestamp else None,
                "event_frequency_per_day": self.historical.event_frequency_per_day,
                "trust_score": self.historical.trust_score,
                "risk_flags": self.historical.risk_flags,
            },
            "entity": {
                "primary_entity_id": self.entity.primary_entity_id,
                "primary_entity_type": self.entity.primary_entity_type,
                "primary_entity_verified": self.entity.primary_entity_verified,
                "secondary_entity_id": self.entity.secondary_entity_id,
                "secondary_entity_type": self.entity.secondary_entity_type,
                "secondary_entity_trust": self.entity.secondary_entity_trust,
                "relationship_age_days": self.entity.relationship_age_days,
            },
            "domain_context": self.domain_context,
            "context_timestamp": self.context_timestamp.isoformat(),
            "context_version": self.context_version,
        }


@dataclass
class UniversalEvent:
    """Universal event structure that works across all domains.

    This abstraction allows the same cascade routing logic to work for:
    - Financial transactions
    - Healthcare claims
    - Content moderation posts
    - Security access attempts
    - Customer support tickets
    - And any other domain

    Attributes:
        id: Unique event identifier
        domain: The domain this event belongs to
        event_type: Type of event (transaction, post, claim, etc.)
        timestamp: When the event occurred
        primary_entity: Who initiated (user, customer, patient, employee)
        secondary_entity: Who received/target (merchant, provider, system)
        value: Numeric value (amount, size, count, severity 0-1)
        unit: Unit of value (USD, bytes, items, severity_score)
        domain_data: Domain-specific payload
    """

    id: str
    domain: DomainType
    event_type: str
    timestamp: datetime

    # Generic entity references
    primary_entity: str  # who initiated
    secondary_entity: str  # target/recipient

    # Generic value (meaning depends on domain)
    value: float
    unit: str  # USD, EUR, bytes, count, severity_score, etc.

    # Domain-specific data
    domain_data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    correlation_id: Optional[str] = None
    source_system: Optional[str] = None
    event_version: str = "1.0"

    def __post_init__(self):
        """Generate IDs if not provided."""
        if not self.id:
            self.id = f"evt_{uuid.uuid4().hex[:12]}"
        if not self.correlation_id:
            self.correlation_id = f"corr_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "domain": self.domain.value,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "primary_entity": self.primary_entity,
            "secondary_entity": self.secondary_entity,
            "value": self.value,
            "unit": self.unit,
            "domain_data": self.domain_data,
            "correlation_id": self.correlation_id,
            "source_system": self.source_system,
            "event_version": self.event_version,
        }


@dataclass
class EventWithContext:
    """Complete event package with context - the input to cascade processing.

    This is the primary input type for the CascadeEngine.execute() method.
    It combines what happened (Event) with the circumstances (Context).

    Example:
        >>> event = UniversalEvent(
        ...     id="evt_123",
        ...     domain=DomainType.CONTENT_MODERATION,
        ...     event_type="post",
        ...     timestamp=datetime.now(),
        ...     primary_entity="user_456",
        ...     secondary_entity="forum_general",
        ...     value=0.0,  # no monetary value
        ...     unit="post",
        ...     domain_data={"content": "Hello world", "has_media": False}
        ... )
        >>> context = EventContext(
        ...     session=SessionContext(ip_address="1.2.3.4"),
        ...     historical=HistoricalContext(account_age_days=30, previous_events_count=100)
        ... )
        >>> event_with_context = EventWithContext(event=event, context=context)
        >>> result = await engine.execute(event_with_context)
    """

    event: UniversalEvent
    context: EventContext

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for cascade processing."""
        return {
            "event": self.event.to_dict(),
            "context": self.context.to_dict(),
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for backward compatibility.

        This allows the event+context to work with existing ExecutionContext
        that expects flat dictionary access like ctx.get("event.value").
        """
        return {
            "event": self.event.to_dict(),
            "context": self.context.to_dict(),
            # Also expose key fields at top level for convenience
            "event_id": self.event.id,
            "domain": self.event.domain.value,
            "event_type": self.event.event_type,
            "value": self.event.value,
            "unit": self.event.unit,
            "primary_entity": self.event.primary_entity,
            "secondary_entity": self.event.secondary_entity,
            "timestamp": self.event.timestamp.isoformat(),
            # Context shortcuts
            "ip_address": self.context.session.ip_address,
            "device_type": self.context.device.device_type,
            "country": self.context.location.country,
            "account_age_days": self.context.historical.account_age_days,
            "trust_score": self.context.historical.trust_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventWithContext":
        """Parse from dictionary (e.g., API request)."""
        event_data = data.get("event", data)  # Support both nested and flat
        context_data = data.get("context", {})

        # Parse event
        event = UniversalEvent(
            id=event_data.get("id", ""),
            domain=DomainType(event_data.get("domain", "GENERIC")),
            event_type=event_data.get("event_type", "generic_event"),
            timestamp=datetime.fromisoformat(event_data["timestamp"]) if "timestamp" in event_data else datetime.now(),
            primary_entity=event_data.get("primary_entity", "unknown"),
            secondary_entity=event_data.get("secondary_entity", "unknown"),
            value=float(event_data.get("value", 0)),
            unit=event_data.get("unit", "unit"),
            domain_data=event_data.get("domain_data", {}),
            correlation_id=event_data.get("correlation_id"),
            source_system=event_data.get("source_system"),
        )

        # Parse context
        session_data = context_data.get("session", {})
        device_data = context_data.get("device", {})
        location_data = context_data.get("location", {})
        historical_data = context_data.get("historical", {})
        entity_data = context_data.get("entity", {})

        context = EventContext(
            session=SessionContext(
                ip_address=session_data.get("ip_address"),
                user_agent=session_data.get("user_agent"),
                session_id=session_data.get("session_id"),
                is_authenticated=session_data.get("is_authenticated"),
            ),
            device=DeviceContext(
                device_id=device_data.get("device_id"),
                device_type=device_data.get("device_type"),
                is_trusted_device=device_data.get("is_trusted_device"),
            ),
            location=LocationContext(
                current_location=location_data.get("current_location"),
                country=location_data.get("country"),
                vpn_detected=location_data.get("vpn_detected"),
            ),
            historical=HistoricalContext(
                account_age_days=historical_data.get("account_age_days"),
                previous_events_count=historical_data.get("previous_events_count"),
                average_event_value=historical_data.get("average_event_value"),
                trust_score=historical_data.get("trust_score"),
                risk_flags=historical_data.get("risk_flags"),
            ),
            entity=EntityContext(
                primary_entity_id=entity_data.get("primary_entity_id"),
                primary_entity_type=entity_data.get("primary_entity_type"),
                secondary_entity_id=entity_data.get("secondary_entity_id"),
            ),
            domain_context=context_data.get("domain_context", {}),
        )

        return cls(event=event, context=context)


# =============================================================================
# Factory functions for common domains
# =============================================================================


def create_finance_event(
    transaction_id: str,
    user_id: str,
    merchant_id: str,
    amount: float,
    currency: str = "USD",
    transaction_type: str = "purchase",
    **domain_data,
) -> UniversalEvent:
    """Create a finance domain event."""
    return UniversalEvent(
        id=transaction_id,
        domain=DomainType.FINANCE,
        event_type=transaction_type,
        timestamp=datetime.now(),
        primary_entity=user_id,
        secondary_entity=merchant_id,
        value=amount,
        unit=currency,
        domain_data={"transaction_type": transaction_type, **domain_data},
    )


def create_content_event(
    content_id: str,
    user_id: str,
    target: str,
    content_type: str = "post",
    content: str = "",
    **domain_data,
) -> UniversalEvent:
    """Create a content moderation domain event."""
    return UniversalEvent(
        id=content_id,
        domain=DomainType.CONTENT_MODERATION,
        event_type=content_type,
        timestamp=datetime.now(),
        primary_entity=user_id,
        secondary_entity=target,
        value=len(content),  # content length as value
        unit="characters",
        domain_data={"content": content, "content_type": content_type, **domain_data},
    )


def create_security_event(
    event_id: str,
    user_id: str,
    resource: str,
    action: str = "access_attempt",
    risk_score: float = 0.0,
    **domain_data,
) -> UniversalEvent:
    """Create a cybersecurity domain event."""
    return UniversalEvent(
        id=event_id,
        domain=DomainType.CYBERSECURITY,
        event_type=action,
        timestamp=datetime.now(),
        primary_entity=user_id,
        secondary_entity=resource,
        value=risk_score,
        unit="risk_score",
        domain_data={"action": action, **domain_data},
    )


def create_support_event(
    ticket_id: str,
    customer_id: str,
    category: str,
    priority: float = 0.5,
    subject: str = "",
    body: str = "",
    **domain_data,
) -> UniversalEvent:
    """Create a customer support domain event."""
    return UniversalEvent(
        id=ticket_id,
        domain=DomainType.CUSTOMER_SUPPORT,
        event_type="ticket",
        timestamp=datetime.now(),
        primary_entity=customer_id,
        secondary_entity=category,
        value=priority,
        unit="priority_score",
        domain_data={"subject": subject, "body": body, **domain_data},
    )
