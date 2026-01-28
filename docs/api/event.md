# Event + Context Module

The event module provides domain-agnostic event and context models for cascade processing. These models work across all domains including finance, healthcare, cybersecurity, content moderation, and more.

## UniversalEvent

::: rotalabs_cascade.core.event.UniversalEvent
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - to_dict

## EventContext

::: rotalabs_cascade.core.event.EventContext
    options:
      show_source: false
      heading_level: 3
      members:
        - to_dict

## EventWithContext

::: rotalabs_cascade.core.event.EventWithContext
    options:
      show_source: false
      heading_level: 3
      members:
        - to_dict
        - to_flat_dict
        - from_dict

## DomainType

::: rotalabs_cascade.core.event.DomainType
    options:
      show_source: false
      heading_level: 3

## SessionContext

::: rotalabs_cascade.core.event.SessionContext
    options:
      show_source: false
      heading_level: 3

## DeviceContext

::: rotalabs_cascade.core.event.DeviceContext
    options:
      show_source: false
      heading_level: 3

## LocationContext

::: rotalabs_cascade.core.event.LocationContext
    options:
      show_source: false
      heading_level: 3

## HistoricalContext

::: rotalabs_cascade.core.event.HistoricalContext
    options:
      show_source: false
      heading_level: 3
