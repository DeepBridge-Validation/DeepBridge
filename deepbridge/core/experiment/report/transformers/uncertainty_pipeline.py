"""
Uncertainty Pipeline - Concrete implementation using Transform Pipeline.

This module demonstrates how to use the Transform Pipeline for uncertainty reports.
Created in Phase 2 Sprint 7-8 as part of TAREFA 7.1.

Example Usage:
    >>> from .uncertainty_pipeline import create_uncertainty_pipeline
    >>>
    >>> # Create pipeline
    >>> pipeline = create_uncertainty_pipeline()
    >>>
    >>> # Execute on raw experiment data
    >>> report_data = pipeline.execute(raw_uncertainty_results)
"""

from typing import Dict, Any, List
import logging

from .pipeline import Validator, Transformer, Enricher, TransformPipeline

logger = logging.getLogger("deepbridge.reports")


# ==================================================================================
# Uncertainty Validation
# ==================================================================================

class UncertaintyValidator(Validator):
    """
    Validates uncertainty experiment data.

    Checks for required fields and data integrity before transformation.
    """

    def validate(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate uncertainty experiment data.

        Required structure:
            - test_results: Dict with primary_model data
            - initial_model_evaluation: Dict with feature_importance

        Args:
            data: Raw experiment results

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for test_results
        if 'test_results' not in data:
            errors.append("Missing 'test_results' key")
        else:
            test_results = data['test_results']

            # Check for primary_model
            if 'primary_model' not in test_results:
                errors.append("Missing 'test_results.primary_model' key")
            else:
                primary = test_results['primary_model']

                # Check for CRQR results
                if 'crqr' not in primary:
                    errors.append("Missing 'test_results.primary_model.crqr' key")
                else:
                    crqr = primary['crqr']

                    # Check for alphas
                    if 'alphas' not in crqr or not crqr['alphas']:
                        errors.append("Missing or empty 'alphas' in CRQR results")

        # Check for initial_model_evaluation
        if 'initial_model_evaluation' not in data:
            errors.append("Missing 'initial_model_evaluation' key")
        else:
            initial_eval = data['initial_model_evaluation']

            # Check for feature importance
            if 'feature_importance' not in initial_eval:
                errors.append("Missing 'feature_importance' in initial_model_evaluation")

        return errors


# ==================================================================================
# Uncertainty Transformation
# ==================================================================================

class UncertaintyTransformer(Transformer):
    """
    Transforms raw uncertainty data to structured format.

    Extracts and normalizes data from experiment results.
    """

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform uncertainty data to structured format.

        Args:
            data: Validated raw experiment data

        Returns:
            Structured uncertainty data
        """
        primary_model = data['test_results']['primary_model']
        crqr = primary_model['crqr']
        initial_eval = data['initial_model_evaluation']

        # Extract alpha-level results
        alphas_data = []
        for alpha_key, alpha_data in crqr['alphas'].items():
            if isinstance(alpha_data, dict):
                alphas_data.append({
                    'alpha': float(alpha_key),
                    'coverage': alpha_data.get('coverage', 0.0),
                    'expected_coverage': alpha_data.get('expected_coverage', 0.0),
                    'avg_width': alpha_data.get('avg_width', 0.0),
                    'calibration_error': abs(
                        alpha_data.get('coverage', 0.0) -
                        alpha_data.get('expected_coverage', 0.0)
                    )
                })

        # Sort by alpha
        alphas_data.sort(key=lambda x: x['alpha'])

        # Extract feature importance
        feature_importance = initial_eval.get('feature_importance', {})

        # Create structured output
        return {
            'model_name': data.get('model_name', 'Model'),
            'model_type': primary_model.get('model_type', 'Unknown'),
            'alphas': alphas_data,
            'feature_importance': feature_importance,
            'metadata': {
                'num_alphas': len(alphas_data),
                'num_features': len(feature_importance)
            }
        }


# ==================================================================================
# Uncertainty Enrichment
# ==================================================================================

class UncertaintyEnricher(Enricher):
    """
    Enriches uncertainty data with derived metrics and summaries.

    Adds quality scores, statistical summaries, and convenience fields.
    """

    def enrich(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich uncertainty data with computed metrics.

        Adds:
            - summary: Aggregate statistics
            - uncertainty_score: Quality metric
            - top_features: Most important features
            - is_well_calibrated: Boolean indicator

        Args:
            data: Transformed uncertainty data

        Returns:
            Enriched data with derived metrics
        """
        alphas = data['alphas']
        feature_importance = data['feature_importance']

        # Calculate summary statistics
        if alphas:
            avg_coverage = sum(a['coverage'] for a in alphas) / len(alphas)
            avg_calibration_error = sum(a['calibration_error'] for a in alphas) / len(alphas)
            avg_width = sum(a['avg_width'] for a in alphas) / len(alphas)

            # Calculate uncertainty score (0-1, higher is better)
            # Based on calibration quality
            uncertainty_score = max(0.0, 1.0 - (avg_calibration_error * 2.0))
        else:
            avg_coverage = 0.0
            avg_calibration_error = 0.0
            avg_width = 0.0
            uncertainty_score = 0.0

        # Create summary
        data['summary'] = {
            'uncertainty_score': round(uncertainty_score, 4),
            'total_alphas': len(alphas),
            'avg_coverage': round(avg_coverage, 4),
            'avg_calibration_error': round(avg_calibration_error, 4),
            'avg_width': round(avg_width, 4),
            'is_well_calibrated': avg_calibration_error < 0.05
        }

        # Top features (sorted by absolute importance)
        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            data['top_features'] = sorted_features[:10]
        else:
            data['top_features'] = []

        # Add convenience fields
        data['features'] = {
            'total': len(feature_importance),
            'top_10': data['top_features']
        }

        logger.debug(
            f"Uncertainty enrichment complete: "
            f"score={uncertainty_score:.3f}, "
            f"alphas={len(alphas)}, "
            f"features={len(feature_importance)}"
        )

        return data


# ==================================================================================
# Pipeline Factory
# ==================================================================================

def create_uncertainty_pipeline() -> TransformPipeline:
    """
    Create configured pipeline for uncertainty reports.

    Returns:
        TransformPipeline: Configured pipeline with validation,
                          transformation, and enrichment stages

    Example:
        >>> pipeline = create_uncertainty_pipeline()
        >>> report_data = pipeline.execute(raw_results)
        >>> print(report_data['summary']['uncertainty_score'])
        0.9234
    """
    return (TransformPipeline()
            .add_stage(UncertaintyValidator())
            .add_stage(UncertaintyTransformer())
            .add_stage(UncertaintyEnricher()))


# ==================================================================================
# Main - Example Usage
# ==================================================================================

if __name__ == "__main__":
    """
    Demonstrate uncertainty pipeline usage.
    """
    print("=" * 80)
    print("Uncertainty Pipeline Example")
    print("=" * 80)

    # Example uncertainty data (simplified)
    example_data = {
        'model_name': 'XGBoost',
        'test_results': {
            'primary_model': {
                'model_type': 'XGBClassifier',
                'crqr': {
                    'alphas': {
                        '0.1': {
                            'coverage': 0.91,
                            'expected_coverage': 0.90,
                            'avg_width': 2.34
                        },
                        '0.2': {
                            'coverage': 0.81,
                            'expected_coverage': 0.80,
                            'avg_width': 1.87
                        }
                    }
                }
            }
        },
        'initial_model_evaluation': {
            'feature_importance': {
                'feature1': 0.45,
                'feature2': 0.32,
                'feature3': 0.15,
                'feature4': 0.08
            }
        }
    }

    # Create and execute pipeline
    pipeline = create_uncertainty_pipeline()

    print(f"\nPipeline: {pipeline}")

    try:
        print("\nExecuting pipeline...")
        result = pipeline.execute(example_data)

        # Show results
        print("\n" + "-" * 80)
        print("Results:")
        print("-" * 80)
        print(f"Model: {result['model_name']} ({result['model_type']})")
        print(f"\nSummary:")
        for key, value in result['summary'].items():
            print(f"  {key}: {value}")

        print(f"\nTop Features ({len(result['top_features'])}):")
        for feature, importance in result['top_features'][:5]:
            print(f"  {feature}: {importance:.4f}")

        print("\n" + "=" * 80)
        print("Uncertainty Pipeline Example Complete")
        print("=" * 80)

    except ValueError as e:
        print(f"\nValidation Error: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        raise
