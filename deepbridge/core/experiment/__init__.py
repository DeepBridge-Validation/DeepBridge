"""
Core experiment module for model validation and testing.
This package provides a standard interface for running experiments on ML models.
"""

import logging
import os

logger = logging.getLogger('deepbridge.core.experiment')

# Everything imported below lives inside this package. A failure here means a
# broken installation, so the ImportError is deliberately left to propagate
# instead of being swallowed by a fallback that hides the real cause.
from deepbridge.core.experiment.dependencies import (
    check_dependencies,
    print_dependency_status,
)
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.core.experiment.interfaces import (
    IExperiment,
    ITestRunner,
    ModelResult,
    TestResult,
)
from deepbridge.core.experiment.manager_factory import ManagerFactory
from deepbridge.core.experiment.model_result import (
    BaseModelResult,
    ClassificationModelResult,
    RegressionModelResult,
    create_model_result,
)
from deepbridge.core.experiment.results import (
    ExperimentResult,
    HyperparameterResult,
    ResilienceResult,
    RobustnessResult,
    UncertaintyResult,
    wrap_results,
)
from deepbridge.core.experiment.runner import TestRunner
from deepbridge.core.experiment.test_result_factory import TestResultFactory
from deepbridge.core.experiment.test_strategies import (
    HyperparameterTestStrategy,
    ResilienceTestStrategy,
    RobustnessTestStrategy,
    TestStrategy,
    TestStrategyFactory,
    UncertaintyTestStrategy,
)

# The report manager pulls in optional third-party rendering dependencies
# (jinja2, matplotlib, seaborn, ...). Report generation is optional, so a
# missing dependency is tolerated - but it is reported loudly, never silently.
try:
    from deepbridge.core.experiment.report.report_manager import ReportManager
except ImportError as exc:
    ReportManager = None
    logger.warning(
        'Report generation is disabled: could not import ReportManager (%s). '
        'Install the reporting extras to enable it: '
        'pip install "deepbridge[reports]"',
        exc,
    )

# Get the base directory of the package
base_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
templates_dir = os.path.join(base_dir, 'templates')

# Only instantiate report_manager if ReportManager was successfully imported
if ReportManager is not None:
    report_manager = ReportManager(templates_dir=templates_dir)
else:
    report_manager = None


(
    all_required_installed,
    missing_required,
    missing_optional,
    version_issues,
) = check_dependencies()

if not all_required_installed:
    logger.warning(
        'Some required dependencies are missing (%s); parts of the experiment '
        'API will not work. Run deepbridge.core.experiment.'
        'print_dependency_status() for details.',
        ', '.join(missing_required) if missing_required else 'unknown',
    )

__all__ = [
    'Experiment',
    'TestRunner',
    'IExperiment',
    'ITestRunner',
    'TestResult',
    'ModelResult',
    'ExperimentResult',
    'RobustnessResult',
    'UncertaintyResult',
    'ResilienceResult',
    'HyperparameterResult',
    'wrap_results',
    'check_dependencies',
    'print_dependency_status',
    'BaseModelResult',
    'ClassificationModelResult',
    'RegressionModelResult',
    'create_model_result',
    'TestResultFactory',
    'TestStrategy',
    'TestStrategyFactory',
    'ManagerFactory',
    'RobustnessTestStrategy',
    'UncertaintyTestStrategy',
    'ResilienceTestStrategy',
    'HyperparameterTestStrategy',
]
