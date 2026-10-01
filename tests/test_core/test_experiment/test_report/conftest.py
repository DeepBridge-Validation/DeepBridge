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

``plotly`` leaks the same way, from ``test_plotly_fallback`` in that same file:
it pops every ``plotly*`` entry out of ``sys.modules`` *before* letting
``fallback_dependencies()`` install the stubs, so when Plotly had not been
imported yet there is nothing to put back and the stubs survive the test.
``_create_plotly_fallback()`` installs four entries -- ``plotly``,
``plotly.graph_objects``, ``plotly.express`` and ``plotly.offline`` -- and all
four have to be repaired. Restoring only ``plotly`` is not enough: the real
``plotly`` package is a package again, so ``from .validator_cache import
ValidatorCache`` succeeds, but ``ValidatorCache.get_validator()`` then runs
``from .graph_objects import Layout``, finds the stub still cached under
``plotly.graph_objects`` and dies with ``ImportError: cannot import name
'Layout'``.
"""

import sys

import jinja2 as _real_jinja2
import plotly as _real_plotly
import plotly.express as _real_plotly_express
import plotly.graph_objects as _real_plotly_go
import plotly.offline as _real_plotly_offline
import pytest

#: Every ``sys.modules`` key that ``_create_plotly_fallback()`` overwrites,
#: mapped to the real module object captured here at collection time.
_REAL_PLOTLY_MODULES = {
    'plotly': _real_plotly,
    'plotly.graph_objects': _real_plotly_go,
    'plotly.express': _real_plotly_express,
    'plotly.offline': _real_plotly_offline,
}


@pytest.fixture(autouse=True)
def real_jinja2():
    """Guarantee the real Jinja2 module is installed for every test here."""
    if sys.modules.get('jinja2') is not _real_jinja2:
        sys.modules['jinja2'] = _real_jinja2
    return _real_jinja2


@pytest.fixture(autouse=True)
def real_plotly():
    """Guarantee the real Plotly modules are installed for every test here.

    Unlike ``real_jinja2`` this one puts back whatever it found afterwards, so
    the repair stays inside this package and no test outside it sees a
    different ``sys.modules`` because of a fixture that ran here.

    This docstring used to claim that leaking the real Plotly forward made
    ``tests/test_validation/test_wrappers/test_uncertainty_suite.py`` and
    ``test_enhanced_uncertainty_suite.py`` stop finishing. That does not
    reproduce: both run to the end with the real Plotly installed (measured
    again here: 55 passed in 36s for the two files together). The save/restore
    stays because not leaking global state out of a fixture is correct on its
    own, not because of that claim.
    """
    previous = {name: sys.modules.get(name) for name in _REAL_PLOTLY_MODULES}
    for name, module in _REAL_PLOTLY_MODULES.items():
        if previous[name] is not module:
            sys.modules[name] = module
    try:
        yield _real_plotly
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
