"""
Simple data transformer for fairness reports.

LEGACY FILE - This file now delegates to the refactored fairness module.
For new code, import directly from:
    from deepbridge.core.experiment.report.transformers.fairness import FairnessDataTransformer
"""

from typing import Dict, Any
import logging

from .fairness import FairnessDataTransformer as RefactoredTransformer

logger = logging.getLogger("deepbridge.reports")


class FairnessDataTransformerSimple:
    """
    Legacy transformer class for fairness reports.

    This class now delegates to the refactored FairnessDataTransformer
    to maintain backward compatibility.

    DEPRECATED: Use FairnessDataTransformer from the fairness module instead.
    """

    def __init__(self):
        """Initialize with refactored transformer."""
        logger.debug("Using refactored FairnessDataTransformer")
        self._transformer = RefactoredTransformer()

    def transform(self, results: Dict[str, Any], model_name: str = "Model") -> Dict[str, Any]:
        """
        Transform raw fairness results into report-ready format.

        This method delegates to the refactored implementation.

        Args:
            results: Dictionary containing fairness analysis results from FairnessSuite
            model_name: Name of the model

        Returns:
            Dictionary with transformed data ready for report rendering
        """
        return self._transformer.transform(results, model_name)
