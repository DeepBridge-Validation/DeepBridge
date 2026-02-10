"""Test configuration for DeepBridge test suite."""

import os
import pytest

# Configure matplotlib to use non-GUI backend for tests
os.environ['MPLBACKEND'] = 'Agg'
