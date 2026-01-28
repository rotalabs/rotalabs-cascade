"""Plugin system for rotalabs-cascade.

This module provides a flexible plugin architecture for extending stage functionality
with capabilities like caching, retries, metrics, and circuit breaking.

Author: Subhadip Mitra <subhadip@rotalabs.ai>
Organization: Rotalabs
"""

from rotalabs_cascade.plugins.builtin import (
    CachePlugin,
    CircuitBreakerPlugin,
    MetricsPlugin,
    PluginFactory,
    PluginRegistry,
    RetryPlugin,
    StagePlugin,
)

__all__ = [
    "StagePlugin",
    "PluginRegistry",
    "CachePlugin",
    "RetryPlugin",
    "MetricsPlugin",
    "CircuitBreakerPlugin",
    "PluginFactory",
]
