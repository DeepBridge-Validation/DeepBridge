"""
Shared fixtures for the report subsystem tests.

Other tests in the suite (``tests/test_core/test_experiment/test_dependencies.py``)
exercise ``_create_jinja2_fallback()``, which installs a stub object into
``sys.modules['jinja2']`` and never puts the real library back. Every module in
this package needs the *real* Jinja2 -- the package declares it as a hard
dependency -- so we repair ``sys.modules`` before each test here instead of
letting the stub leak in and blow up with
``AttributeError: 'Jinja2Mock' object has no attribute 'FileSystemLoader'``.

The real module object is captured at collection time (before any test runs) and
put back as-is. It is deliberately *not* re-imported: a fresh import would create
a second copy of ``jinja2`` whose internal ``missing`` sentinel no longer matches
the one captured by templates that were compiled against the first copy.
"""

import sys

import jinja2 as _real_jinja2
import pytest


@pytest.fixture(autouse=True)
def real_jinja2():
    """Guarantee the real Jinja2 module is installed for every test here."""
    if sys.modules.get('jinja2') is not _real_jinja2:
        sys.modules['jinja2'] = _real_jinja2
    return _real_jinja2
