# Event + Context Pattern

The Event + Context pattern is the foundational abstraction that makes rotalabs-cascade domain-agnostic. It cleanly separates **what happened** (the Event) from **the circumstances surrounding it** (the Context), enabling the same cascade routing logic to work across finance, healthcare, cybersecurity, content moderation, and any other domain.

## Overview

Traditional detection systems are tightly coupled to their domain. A fraud detection system speaks in "transactions" and "amounts," while a content moderation system speaks in "posts" and "violations." This coupling makes it difficult to share infrastructure, patterns, and learnings across domains.

The Event + Context pattern solves this by providing universal abstractions:

- **UniversalEvent**: Describes what happened in domain-neutral terms
- **EventContext**: Captures the circumstances (who, where, when, history)
- **EventWithContext**: Combines both as input to the cascade engine

## UniversalEvent Structure

The `UniversalEvent` dataclass represents any action or occurrence that needs to be evaluated by the cascade:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

@dataclass
class UniversalEvent:
    id: str                      # Unique event identifier
    domain: DomainType           # Domain this event belongs to
    event_type: str              # Type of event (transaction, post, claim, etc.)
    timestamp: datetime          # When the event occurred
    primary_entity: str          # Who initiated (user, customer, patient)
    secondary_entity: str        # Target/recipient (merchant, provider, system)
    value: float                 # Numeric value (amount, size, severity 0-1)
    unit: str                    # Unit of value (USD, bytes, severity_score)
    domain_data: Dict[str, Any]  # Domain-specific payload
    correlation_id: Optional[str] = None
    source_system: Optional[str] = None
    event_version: str = "1.0"
```

### Field Semantics by Domain

| Field | Finance | Content Moderation | Cybersecurity | Healthcare |
|-------|---------|-------------------|---------------|------------|
| `primary_entity` | user_id | author_id | user_id | patient_id |
| `secondary_entity` | merchant_id | forum/channel | resource | provider_id |
| `value` | amount | content_length | risk_score | claim_amount |
| `unit` | USD/EUR | characters | risk_score | USD |
| `domain_data` | card_type, mcc | content, has_media | action, ip | diagnosis, codes |

### Supported Domains

The `DomainType` enum defines the supported domains:

```python
class DomainType(Enum):
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
```

## EventContext Structure

The `EventContext` dataclass captures contextual information surrounding an event:

```python
@dataclass
class EventContext:
    session: SessionContext           # Session-related info
    device: DeviceContext             # Device-related info
    location: LocationContext         # Location-related info
    historical: HistoricalContext     # Historical behavior
    entity: EntityContext             # Entity information
    domain_context: Dict[str, Any]    # Domain-specific extension
    context_timestamp: datetime       # When context was captured
    context_version: str = "1.0"
```

### SessionContext

Captures session and authentication information:

```python
@dataclass
class SessionContext:
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    login_timestamp: Optional[datetime] = None
    session_duration_minutes: Optional[int] = None
    previous_session_count: Optional[int] = None
    is_authenticated: Optional[bool] = None
    auth_method: Optional[str] = None  # password, oauth, sso, mfa
```

### DeviceContext

Captures device fingerprint and characteristics:

```python
@dataclass
class DeviceContext:
    device_id: Optional[str] = None
    device_type: Optional[str] = None  # mobile, desktop, tablet, iot
    device_info: Optional[str] = None
    operating_system: Optional[str] = None
    browser: Optional[str] = None
    app_version: Optional[str] = None
    is_trusted_device: Optional[bool] = None
    device_fingerprint: Optional[str] = None
```

### LocationContext

Captures geographic and network location:

```python
@dataclass
class LocationContext:
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
```

### HistoricalContext

Captures behavioral history and trust signals:

```python
@dataclass
class HistoricalContext:
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
```

### EntityContext

Captures information about the entities involved:

```python
@dataclass
class EntityContext:
    primary_entity_id: Optional[str] = None
    primary_entity_type: Optional[str] = None  # user, customer, patient
    primary_entity_verified: Optional[bool] = None
    secondary_entity_id: Optional[str] = None
    secondary_entity_type: Optional[str] = None  # merchant, provider, vendor
    secondary_entity_trust: Optional[float] = None
    relationship_age_days: Optional[int] = None
```

## EventWithContext

The `EventWithContext` dataclass combines both structures and serves as the primary input to the cascade engine:

```python
@dataclass
class EventWithContext:
    event: UniversalEvent
    context: EventContext
```

### Example Usage

```python
from rotalabs_cascade import (
    UniversalEvent, EventContext, EventWithContext,
    DomainType, SessionContext, HistoricalContext, CascadeEngine
)
from datetime import datetime

# Create the event
event = UniversalEvent(
    id="evt_abc123",
    domain=DomainType.CONTENT_MODERATION,
    event_type="post",
    timestamp=datetime.now(),
    primary_entity="user_456",
    secondary_entity="forum_general",
    value=0.0,
    unit="post",
    domain_data={
        "content": "Check out this amazing deal!",
        "has_media": False,
        "mentions": ["@everyone"]
    }
)

# Create the context
context = EventContext(
    session=SessionContext(
        ip_address="192.168.1.100",
        is_authenticated=True
    ),
    historical=HistoricalContext(
        account_age_days=30,
        previous_events_count=100,
        trust_score=0.85
    )
)

# Combine into EventWithContext
event_with_context = EventWithContext(event=event, context=context)

# Execute through cascade
result = await engine.execute(event_with_context)
```

## Domain Examples

### Finance: Transaction Fraud Detection

```python
from rotalabs_cascade import create_finance_event, EventContext, EventWithContext

# Use the factory function for finance events
event = create_finance_event(
    transaction_id="txn_789",
    user_id="cust_123",
    merchant_id="merch_456",
    amount=2500.00,
    currency="USD",
    transaction_type="purchase",
    # Additional domain data
    card_type="credit",
    mcc_code="5411",  # Grocery stores
    is_card_present=False
)

context = EventContext(
    session=SessionContext(
        ip_address="203.0.113.50",
        auth_method="mfa"
    ),
    device=DeviceContext(
        device_type="mobile",
        is_trusted_device=True
    ),
    location=LocationContext(
        country="US",
        region="CA",
        vpn_detected=False
    ),
    historical=HistoricalContext(
        account_age_days=730,
        average_event_value=150.00,
        max_event_value=500.00,
        trust_score=0.92
    )
)

event_with_context = EventWithContext(event=event, context=context)
```

### Content Moderation: Post Review

```python
from rotalabs_cascade import create_content_event, EventContext, EventWithContext

# Use the factory function for content events
event = create_content_event(
    content_id="post_abc123",
    user_id="author_456",
    target="community_gaming",
    content_type="post",
    content="Join my Discord for free giveaways! Click here: bit.ly/xxx",
    # Additional domain data
    has_links=True,
    link_count=1,
    mentions_count=0,
    hashtags=["giveaway", "free"]
)

context = EventContext(
    session=SessionContext(
        ip_address="198.51.100.25",
        user_agent="Mozilla/5.0..."
    ),
    historical=HistoricalContext(
        account_age_days=3,
        previous_events_count=5,
        trust_score=0.2,
        risk_flags=["new_account", "rapid_posting"]
    )
)

event_with_context = EventWithContext(event=event, context=context)
```

### Cybersecurity: Access Attempt

```python
from rotalabs_cascade import create_security_event, EventContext, EventWithContext

# Use the factory function for security events
event = create_security_event(
    event_id="sec_xyz789",
    user_id="employee_123",
    resource="database_prod",
    action="access_attempt",
    risk_score=0.7,
    # Additional domain data
    requested_permission="read",
    time_of_day="02:30",
    is_after_hours=True
)

context = EventContext(
    session=SessionContext(
        ip_address="10.0.0.50",
        auth_method="sso"
    ),
    device=DeviceContext(
        device_type="desktop",
        is_trusted_device=False,
        device_fingerprint="unknown"
    ),
    location=LocationContext(
        country="CN",
        vpn_detected=True
    ),
    historical=HistoricalContext(
        typical_event_times=["09:00-17:00"],
        typical_locations=["US"],
        trust_score=0.6
    )
)

event_with_context = EventWithContext(event=event, context=context)
```

### Healthcare: Claim Review

```python
from rotalabs_cascade import UniversalEvent, DomainType, EventContext, EventWithContext
from datetime import datetime

event = UniversalEvent(
    id="claim_12345",
    domain=DomainType.HEALTHCARE,
    event_type="claim",
    timestamp=datetime.now(),
    primary_entity="patient_789",
    secondary_entity="provider_456",
    value=15000.00,
    unit="USD",
    domain_data={
        "diagnosis_codes": ["J18.9", "R05"],
        "procedure_codes": ["99213", "94640"],
        "service_date": "2024-01-15",
        "facility_type": "outpatient"
    }
)

context = EventContext(
    entity=EntityContext(
        primary_entity_type="patient",
        secondary_entity_type="provider",
        secondary_entity_trust=0.95,
        relationship_age_days=1825  # 5 years
    ),
    historical=HistoricalContext(
        average_event_value=500.00,
        max_event_value=3000.00,
        previous_events_count=45
    ),
    domain_context={
        "provider_specialty": "pulmonology",
        "patient_insurance_type": "PPO",
        "pre_authorization": True
    }
)

event_with_context = EventWithContext(event=event, context=context)
```

## Why Domain-Agnostic?

The Event + Context pattern enables several powerful capabilities:

### 1. Shared Infrastructure

The same cascade engine, routing rules, and execution infrastructure work across all domains. You build it once and apply it everywhere.

### 2. Pattern Transfer

Patterns learned in one domain can inform detection in another. A velocity check pattern in fraud detection (too many transactions too fast) maps naturally to spam detection (too many posts too fast).

### 3. Unified Observability

Metrics, logging, and monitoring use the same schema regardless of domain. A single dashboard can show detection rates, stage latencies, and cost metrics across your entire organization.

### 4. Consistent Developer Experience

Teams working on different domains use the same APIs, configuration format, and mental model. Knowledge transfers easily between teams.

### 5. Flexible Extension

The `domain_data` field in `UniversalEvent` and `domain_context` field in `EventContext` provide escape hatches for domain-specific data without breaking the universal schema.

## Converting from Flat Dictionaries

For backward compatibility, `EventWithContext` can be converted to a flat dictionary:

```python
event_with_context = EventWithContext(event=event, context=context)

# Get flat dictionary for ExecutionContext
flat_dict = event_with_context.to_flat_dict()

# Contains both nested structure and top-level shortcuts:
# {
#     "event": {...},
#     "context": {...},
#     "event_id": "evt_abc123",
#     "domain": "CONTENT_MODERATION",
#     "value": 0.0,
#     "ip_address": "192.168.1.100",
#     "account_age_days": 30,
#     ...
# }
```

## Parsing from API Requests

When receiving events from external systems, use `from_dict`:

```python
# Incoming API request
request_data = {
    "event": {
        "id": "evt_123",
        "domain": "FINANCE",
        "event_type": "transaction",
        "timestamp": "2024-01-15T10:30:00",
        "primary_entity": "user_456",
        "secondary_entity": "merchant_789",
        "value": 100.00,
        "unit": "USD",
        "domain_data": {"card_type": "debit"}
    },
    "context": {
        "session": {"ip_address": "192.168.1.1"},
        "historical": {"account_age_days": 365}
    }
}

# Parse into strongly-typed objects
event_with_context = EventWithContext.from_dict(request_data)
```

## Summary

The Event + Context pattern is the foundation that makes rotalabs-cascade truly domain-agnostic:

- **UniversalEvent** captures what happened using generic fields that map to any domain
- **EventContext** captures the circumstances through structured sub-contexts
- **EventWithContext** combines both as the primary input to cascade processing
- Factory functions (`create_finance_event`, `create_content_event`, etc.) simplify event creation
- The pattern enables shared infrastructure, pattern transfer, and consistent developer experience
