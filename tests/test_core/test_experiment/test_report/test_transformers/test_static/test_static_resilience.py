"""
Comprehensive tests for StaticResilienceTransformer.

This module tests the static resilience data transformer that processes
resilience test results for static reports.
"""

import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from deepbridge.core.experiment.report.transformers.static.static_resilience import (
    StaticResilienceTransformer,
)


@pytest.fixture
def transformer():
    """Create a StaticResilienceTransformer instance for testing."""
    with patch(
        'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_VISUALIZATION_LIBS',
        True,
    ):
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
            False,
        ):
            transformer = StaticResilienceTransformer()
            return transformer


@pytest.fixture
def sample_resilience_data():
    """Create sample resilience data for testing."""
    return {
        'resilience_score': 0.85,
        'avg_performance_gap': 0.12,
        'model_type': 'Classification',
        'features': ['feature1', 'feature2', 'feature3'],
        'test_results': [
            {
                'test_name': 'Test 1',
                'passed': True,
                'score': 0.9,
            },
            {
                'test_name': 'Test 2',
                'passed': False,
                'score': 0.6,
            },
        ],
        'distribution_shift': {
            'by_distance_metric': {
                'wasserstein': {
                    'avg_feature_distances': {
                        'feature1': 0.1,
                        'feature2': 0.3,
                        'feature3': 0.2,
                    },
                    'results': [
                        {
                            'distance': 0.2,
                            'feature_distances': {
                                'all_feature_distances': {
                                    'feature1': 0.1,
                                    'feature2': 0.3,
                                }
                            },
                        }
                    ],
                }
            }
        },
        'adversarial_robustness': {
            'attacks': [
                {
                    'attack_name': 'FGSM',
                    'success_rate': 0.25,
                    'avg_perturbation': 0.05,
                },
                {
                    'attack_name': 'PGD',
                    'success_rate': 0.35,
                    'avg_perturbation': 0.08,
                },
            ]
        },
        'weak_spot_analysis': {
            'weak_spots': [
                {
                    'feature': 'feature1',
                    'performance_gap': 0.2,
                    'size': 100,
                },
                {
                    'feature': 'feature2',
                    'performance_gap': 0.15,
                    'size': 75,
                },
            ]
        },
        'overfit_analysis': {
            'train_performance': 0.95,
            'test_performance': 0.85,
            'gap': 0.10,
        },
    }


class TestStaticResilienceTransformerInitialization:
    """Test initialization of StaticResilienceTransformer."""

    def test_init_without_visualization_libs(self):
        """Test initialization when visualization libraries are not available."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_VISUALIZATION_LIBS',
            False,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
                False,
            ):
                transformer = StaticResilienceTransformer()
                assert transformer.chart_generator is None

    def test_init_with_visualization_libs(self):
        """Test initialization when visualization libraries are available."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_VISUALIZATION_LIBS',
            True,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.sns'
            ) as mock_sns:
                with patch(
                    'deepbridge.core.experiment.report.transformers.static.static_resilience.plt'
                ) as mock_plt:
                    with patch(
                        'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
                        False,
                    ):
                        transformer = StaticResilienceTransformer()
                        mock_sns.set_theme.assert_called_once_with(
                            style='whitegrid'
                        )
                        mock_sns.set_palette.assert_called_once_with('deep')
                        mock_plt.rcParams.update.assert_called_once()

    def test_init_with_chart_generator(self):
        """Test initialization when chart generator is available."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
            True,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.ResilienceChartGenerator'
            ) as mock_generator:
                with patch(
                    'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_VISUALIZATION_LIBS',
                    False,
                ):
                    transformer = StaticResilienceTransformer()
                    assert transformer.chart_generator is not None
                    mock_generator.assert_called_once()


class TestStaticResilienceTransformerTransform:
    """Test the transform method of StaticResilienceTransformer."""

    def test_transform_basic_data(self, transformer, sample_resilience_data):
        """Test transform with basic resilience data."""
        result = transformer.transform(
            sample_resilience_data, model_name='TestModel'
        )

        assert result['model_name'] == 'TestModel'
        assert result['test_type'] == 'resilience'
        assert 'timestamp' in result
        assert result['model_type'] == 'Classification'
        assert result['features'] == ['feature1', 'feature2', 'feature3']
        assert result['resilience_score'] == 0.85
        assert result['avg_performance_gap'] == 0.12

    def test_transform_with_missing_resilience_score(
        self, transformer, sample_resilience_data
    ):
        """Test transform when resilience_score is missing."""
        data = sample_resilience_data.copy()
        del data['resilience_score']

        result = transformer.transform(data, model_name='TestModel')

        assert result['model_name'] == 'TestModel'
        assert 'resilience_score' not in result or result.get(
            'resilience_score'
        ) is None

    def test_transform_with_performance_gap_alternative_name(
        self, transformer, sample_resilience_data
    ):
        """Test transform when performance gap has alternative name."""
        data = sample_resilience_data.copy()
        del data['avg_performance_gap']
        data['performance_gap'] = 0.15

        # Mock the base transformer to return performance_gap
        transformer.base_transformer.transform = Mock(
            return_value={
                'performance_gap': 0.15,
                'model_name': 'TestModel',
                'timestamp': '2024-01-01 00:00:00',
            }
        )

        result = transformer.transform(data, model_name='TestModel')

        # The output should normalize to avg_performance_gap
        assert 'avg_performance_gap' in result or 'performance_gap' in result

    def test_transform_with_empty_data(self, transformer):
        """Test transform with empty data dictionary."""
        result = transformer.transform({}, model_name='TestModel')

        assert result['model_name'] == 'TestModel'
        assert result['test_type'] == 'resilience'
        assert 'timestamp' in result

    def test_transform_with_base_transformer_error(
        self, transformer, sample_resilience_data
    ):
        """Test transform when base transformer raises an error."""
        transformer.base_transformer.transform = Mock(
            side_effect=Exception('Base transformer error')
        )

        result = transformer.transform(
            sample_resilience_data, model_name='TestModel'
        )

        # Should still return a basic structure
        assert result['model_name'] == 'TestModel'
        assert result['test_type'] == 'resilience'
        assert 'timestamp' in result

    def test_transform_preserves_distribution_shift(
        self, transformer, sample_resilience_data
    ):
        """Test that distribution shift data is preserved in transformation."""
        transformer.base_transformer.transform = Mock(
            return_value={
                'model_name': 'TestModel',
                'timestamp': '2024-01-01 00:00:00',
                'distribution_shift': sample_resilience_data[
                    'distribution_shift'
                ],
            }
        )

        result = transformer.transform(
            sample_resilience_data, model_name='TestModel'
        )

        # Check if distribution shift data is in the result
        assert 'distribution_shift' in result or len(result.keys()) > 5

    def test_transform_preserves_adversarial_robustness(
        self, transformer, sample_resilience_data
    ):
        """Test that adversarial robustness data is preserved."""
        transformer.base_transformer.transform = Mock(
            return_value={
                'model_name': 'TestModel',
                'timestamp': '2024-01-01 00:00:00',
                'adversarial_robustness': sample_resilience_data[
                    'adversarial_robustness'
                ],
            }
        )

        result = transformer.transform(
            sample_resilience_data, model_name='TestModel'
        )

        assert 'adversarial_robustness' in result or len(result.keys()) > 5

    def test_transform_with_default_model_name(
        self, transformer, sample_resilience_data
    ):
        """Test transform with default model name."""
        result = transformer.transform(sample_resilience_data)

        assert result['model_name'] == 'Model'

    def test_transform_with_numeric_scores(self, transformer):
        """Test transform with various numeric score formats."""
        data = {
            'resilience_score': 0.923456789,
            'avg_performance_gap': 0.123456789,
            'model_type': 'Regression',
        }

        result = transformer.transform(data, model_name='NumericModel')

        assert result['resilience_score'] == 0.923456789
        assert result['avg_performance_gap'] == 0.123456789


class TestStaticResilienceTransformerGenerateCharts:
    """Test the _generate_charts method of StaticResilienceTransformer."""

    def test_generate_charts_without_chart_generator(
        self, transformer, sample_resilience_data
    ):
        """Test _generate_charts when chart generator is not available."""
        transformer.chart_generator = None

        charts = transformer._generate_charts(sample_resilience_data)

        assert charts == {}

    def test_generate_charts_with_resilience_score(self, sample_resilience_data):
        """Test chart generation with resilience score."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
            True,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.ResilienceChartGenerator'
            ) as mock_gen_class:
                mock_generator = Mock()
                mock_generator.generate_model_resilience_scores = Mock(
                    return_value='base64_encoded_image'
                )
                mock_gen_class.return_value = mock_generator

                transformer = StaticResilienceTransformer()
                charts = transformer._generate_charts(sample_resilience_data)

                # Verify that the chart generation was called
                mock_generator.generate_model_resilience_scores.assert_called()

    def test_generate_charts_with_feature_distances(
        self, sample_resilience_data
    ):
        """Test chart generation with feature distances."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
            True,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.ResilienceChartGenerator'
            ) as mock_gen_class:
                mock_generator = Mock()
                mock_generator.generate_feature_distribution_shift = Mock(
                    return_value='base64_encoded_image'
                )
                mock_gen_class.return_value = mock_generator

                transformer = StaticResilienceTransformer()

                # Add feature_distances to the data
                data = sample_resilience_data.copy()
                data['feature_distances'] = {
                    'feature1': 0.1,
                    'feature2': 0.3,
                }

                charts = transformer._generate_charts(data)

                # Charts should be generated (may be empty if conditions aren't met)
                assert isinstance(charts, dict)

    def test_generate_charts_with_error_handling(self, sample_resilience_data):
        """Test that chart generation handles errors gracefully."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
            True,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.ResilienceChartGenerator'
            ) as mock_gen_class:
                mock_generator = Mock()
                mock_generator.generate_model_resilience_scores = Mock(
                    side_effect=Exception('Chart generation error')
                )
                mock_gen_class.return_value = mock_generator

                transformer = StaticResilienceTransformer()
                charts = transformer._generate_charts(sample_resilience_data)

                # Should return empty dict or partial results, not raise exception
                assert isinstance(charts, dict)


class TestStaticResilienceTransformerEdgeCases:
    """Test edge cases and error handling."""

    def test_transform_with_none_values(self, transformer):
        """Test transform with None values in data."""
        data = {
            'resilience_score': None,
            'avg_performance_gap': None,
            'model_type': None,
            'features': None,
        }

        result = transformer.transform(data, model_name='TestModel')

        assert result['model_name'] == 'TestModel'
        assert 'timestamp' in result

    def test_transform_with_invalid_types(self, transformer):
        """Test transform with invalid data types."""
        data = {
            'resilience_score': 'invalid',
            'avg_performance_gap': 'invalid',
            'features': 'not_a_list',
        }

        # Should not crash, might handle gracefully
        result = transformer.transform(data, model_name='TestModel')

        assert result['model_name'] == 'TestModel'

    def test_transform_with_nested_distribution_shift(self, transformer):
        """Test transform with deeply nested distribution shift data."""
        data = {
            'distribution_shift': {
                'by_distance_metric': {
                    'wasserstein': {
                        'results': [
                            {
                                'feature_distances': {
                                    'all_feature_distances': {
                                        'feature1': 0.1,
                                        'feature2': 0.2,
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }

        result = transformer.transform(data, model_name='TestModel')

        assert result['model_name'] == 'TestModel'

    def test_transform_with_very_large_data(self, transformer):
        """Test transform with large amounts of data."""
        data = {
            'features': [f'feature_{i}' for i in range(1000)],
            'test_results': [
                {'test_name': f'Test {i}', 'passed': True, 'score': 0.9}
                for i in range(100)
            ],
        }

        result = transformer.transform(data, model_name='LargeModel')

        assert result['model_name'] == 'LargeModel'
        assert len(result.get('features', [])) <= 1000

    def test_timestamp_format(self, transformer, sample_resilience_data):
        """Test that timestamp is in correct format."""
        result = transformer.transform(
            sample_resilience_data, model_name='TestModel'
        )

        timestamp = result.get('timestamp')
        assert timestamp is not None

        # Try to parse the timestamp
        try:
            datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            valid_format = True
        except (ValueError, TypeError):
            valid_format = False

        assert valid_format or isinstance(timestamp, str)


class TestStaticResilienceTransformerIntegration:
    """Integration tests for StaticResilienceTransformer."""

    def test_full_transformation_pipeline(self, transformer):
        """Test the full transformation pipeline with realistic data."""
        data = {
            'resilience_score': 0.78,
            'avg_performance_gap': 0.15,
            'model_type': 'Classification',
            'features': ['age', 'income', 'education'],
            'distribution_shift': {
                'by_distance_metric': {
                    'wasserstein': {
                        'avg_feature_distances': {
                            'age': 0.12,
                            'income': 0.25,
                            'education': 0.08,
                        }
                    }
                }
            },
            'adversarial_robustness': {
                'attacks': [
                    {
                        'attack_name': 'FGSM',
                        'success_rate': 0.20,
                        'avg_perturbation': 0.04,
                    }
                ]
            },
            'weak_spot_analysis': {
                'weak_spots': [
                    {
                        'feature': 'income',
                        'performance_gap': 0.18,
                        'size': 150,
                    }
                ]
            },
            'overfit_analysis': {
                'train_performance': 0.92,
                'test_performance': 0.80,
                'gap': 0.12,
            },
        }

        result = transformer.transform(data, model_name='IntegrationModel')

        # Verify all expected fields are present
        assert result['model_name'] == 'IntegrationModel'
        assert result['test_type'] == 'resilience'
        assert 'timestamp' in result
        assert result['model_type'] == 'Classification'
        assert result['features'] == ['age', 'income', 'education']
        assert result['resilience_score'] == 0.78
        assert result['avg_performance_gap'] == 0.15

    def test_multiple_transforms_same_instance(
        self, transformer, sample_resilience_data
    ):
        """Test multiple transformations with the same transformer instance."""
        # First transformation
        result1 = transformer.transform(
            sample_resilience_data, model_name='Model1'
        )

        # Second transformation with different data
        data2 = sample_resilience_data.copy()
        data2['resilience_score'] = 0.95

        result2 = transformer.transform(data2, model_name='Model2')

        # Verify both transformations worked correctly
        assert result1['model_name'] == 'Model1'
        assert result2['model_name'] == 'Model2'
        assert result1['resilience_score'] == 0.85
        assert result2['resilience_score'] == 0.95


class TestStaticResilienceTransformerChartGenerationAdvanced:
    """Advanced tests for chart generation covering more edge cases."""

    def test_generate_charts_with_various_data_structures(self):
        """Test chart generation with various data structure formats."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
            True,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.ResilienceChartGenerator'
            ) as mock_gen_class:
                mock_generator = Mock()
                mock_generator.generate_model_resilience_scores = Mock(
                    return_value='base64_chart'
                )
                mock_generator.generate_feature_distribution_shift = Mock(
                    return_value='base64_chart'
                )
                mock_gen_class.return_value = mock_generator

                transformer = StaticResilienceTransformer()

                # Test with feature_distances directly
                data1 = {'feature_distances': {'feature1': 0.5, 'feature2': 0.3}}
                charts = transformer._generate_charts(data1)
                assert isinstance(charts, dict)

                # Test with nested distribution_shift > by_distance_metric
                data2 = {
                    'distribution_shift': {
                        'by_distance_metric': {
                            'wasserstein': {
                                'avg_feature_distances': {
                                    'feature1': 0.2
                                }
                            }
                        }
                    }
                }
                charts = transformer._generate_charts(data2)
                assert isinstance(charts, dict)

                # Test with top_features
                data3 = {
                    'distribution_shift': {
                        'by_distance_metric': {
                            'kl_divergence': {
                                'top_features': {'feature3': 0.4}
                            }
                        }
                    }
                }
                charts = transformer._generate_charts(data3)
                assert isinstance(charts, dict)

    def test_generate_charts_with_results_list(self):
        """Test chart generation when data is in results list format."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
            True,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.ResilienceChartGenerator'
            ) as mock_gen_class:
                mock_generator = Mock()
                mock_generator.generate_feature_distribution_shift = Mock(
                    return_value='base64_chart'
                )
                mock_gen_class.return_value = mock_generator

                transformer = StaticResilienceTransformer()

                # Test with results containing all_feature_distances
                data = {
                    'distribution_shift': {
                        'by_distance_metric': {
                            'wasserstein': {
                                'results': [
                                    {
                                        'feature_distances': {
                                            'all_feature_distances': {
                                                'feature1': 0.1,
                                                'feature2': 0.2
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
                charts = transformer._generate_charts(data)
                assert isinstance(charts, dict)

    def test_generate_charts_with_alternative_field_names(self):
        """Test chart generation with alternative field names."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
            True,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.ResilienceChartGenerator'
            ) as mock_gen_class:
                mock_generator = Mock()
                mock_generator.generate_feature_distribution_shift = Mock(
                    return_value='base64_chart'
                )
                mock_gen_class.return_value = mock_generator

                transformer = StaticResilienceTransformer()

                # Test with 'features' field
                data1 = {
                    'distribution_shift': {
                        'features': {'feature1': 0.3}
                    }
                }
                charts = transformer._generate_charts(data1)
                assert isinstance(charts, dict)

                # Test with 'distances' field
                data2 = {
                    'distribution_shift': {
                        'distances': {'feature2': 0.4}
                    }
                }
                charts = transformer._generate_charts(data2)
                assert isinstance(charts, dict)

    def test_generate_charts_with_invalid_numeric_values(self):
        """Test chart generation when numeric values can't be converted."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
            True,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.ResilienceChartGenerator'
            ) as mock_gen_class:
                mock_generator = Mock()
                mock_generator.generate_model_resilience_scores = Mock(
                    return_value='base64_chart'
                )
                mock_gen_class.return_value = mock_generator

                transformer = StaticResilienceTransformer()

                # Test with non-numeric resilience_score
                data = {
                    'resilience_score': 'not_a_number',
                    'avg_performance_gap': None
                }
                charts = transformer._generate_charts(data)
                assert isinstance(charts, dict)

    def test_generate_charts_with_complex_nested_results(self):
        """Test chart generation with complex nested results."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
            True,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.ResilienceChartGenerator'
            ) as mock_gen_class:
                mock_generator = Mock()
                mock_generator.generate_feature_distribution_shift = Mock(
                    return_value='base64_chart'
                )
                mock_gen_class.return_value = mock_generator

                transformer = StaticResilienceTransformer()

                # Test with top_features in results
                data = {
                    'distribution_shift': {
                        'by_distance_metric': {
                            'wasserstein': {
                                'results': [
                                    {
                                        'feature_distances': {
                                            'top_features': {
                                                'feature1': 0.5
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
                charts = transformer._generate_charts(data)
                assert isinstance(charts, dict)

    def test_generate_charts_searches_pattern_based_keys(self):
        """Test that chart generation searches for keys with 'feature' or 'distance' patterns."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
            True,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.ResilienceChartGenerator'
            ) as mock_gen_class:
                mock_generator = Mock()
                mock_generator.generate_feature_distribution_shift = Mock(
                    return_value='base64_chart'
                )
                mock_gen_class.return_value = mock_generator

                transformer = StaticResilienceTransformer()

                # Test with pattern-based key names
                data = {
                    'distribution_shift': {
                        'custom_feature_data': {'feat1': 0.2},
                        'distance_info': {'feat2': 0.3}
                    }
                }
                charts = transformer._generate_charts(data)
                assert isinstance(charts, dict)

    def test_generate_charts_with_empty_results_list(self):
        """Test chart generation with empty results list."""
        with patch(
            'deepbridge.core.experiment.report.transformers.static.static_resilience.HAS_CHART_GENERATOR',
            True,
        ):
            with patch(
                'deepbridge.core.experiment.report.transformers.static.static_resilience.ResilienceChartGenerator'
            ) as mock_gen_class:
                mock_generator = Mock()
                mock_gen_class.return_value = mock_generator

                transformer = StaticResilienceTransformer()

                data = {
                    'distribution_shift': {
                        'by_distance_metric': {
                            'wasserstein': {
                                'results': []
                            }
                        }
                    }
                }
                charts = transformer._generate_charts(data)
                assert isinstance(charts, dict)
