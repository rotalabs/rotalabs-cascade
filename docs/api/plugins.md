# Plugins Module

The plugins module provides built-in plugin implementations for extending cascade stage handlers with caching, retry logic, metrics collection, and circuit breaking.

## CachePlugin

::: rotalabs_cascade.plugins.builtin.CachePlugin
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - name
        - execute

## RetryPlugin

::: rotalabs_cascade.plugins.builtin.RetryPlugin
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - name
        - execute

## MetricsPlugin

::: rotalabs_cascade.plugins.builtin.MetricsPlugin
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - name
        - success_rate
        - avg_time_ms
        - metrics
        - execute

## CircuitBreakerPlugin

::: rotalabs_cascade.plugins.builtin.CircuitBreakerPlugin
    options:
      show_source: false
      heading_level: 3
      members:
        - __init__
        - name
        - execute

## PluginFactory

::: rotalabs_cascade.plugins.builtin.PluginFactory
    options:
      show_source: false
      heading_level: 3
      members:
        - wrap_handler
