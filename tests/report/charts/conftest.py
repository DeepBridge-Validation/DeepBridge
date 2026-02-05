"""
Pytest configuration for chart tests.

Ensures chart registry is properly initialized for all chart tests.
"""

import importlib

import pytest


@pytest.fixture(scope="function", autouse=True)
def reset_and_populate_chart_registry():
    """
    Reset and populate chart registry before each test.

    This ensures test isolation even when other tests clear the registry.
    """
    from deepbridge.core.experiment.report.charts import (
        ChartRegistry,
        examples,
        report_charts,
    )

    # Clear to ensure clean state
    ChartRegistry.clear()

    # Re-import modules to trigger @register_chart decorators
    importlib.reload(examples)
    importlib.reload(report_charts)

    # Also call register functions for non-decorated charts
    examples.register_example_charts()
    report_charts.register_report_charts()

    yield

    # Don't clear after - leave charts for next test that might need them
