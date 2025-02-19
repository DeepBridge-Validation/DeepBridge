# Import components to make them available through the auto module
from deepbridge.auto.config import DistillationConfig
from deepbridge.auto.experiment_runner import ExperimentRunner
from deepbridge.auto.metrics import MetricsEvaluator
from deepbridge.auto.visualization import Visualizer
from deepbridge.auto.reporting import ReportGenerator

# Only expose components that might be directly useful to advanced users
__all__ = ['DistillationConfig', 'MetricsEvaluator', 'Visualizer', 'ReportGenerator']