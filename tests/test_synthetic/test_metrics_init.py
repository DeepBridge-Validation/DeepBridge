"""
Test imports from synthetic.metrics module.
"""

def test_metrics_imports():
    """Test that metrics module imports work"""
    from deepbridge.synthetic.metrics import (
        SyntheticMetrics,
        evaluate_synthetic_quality,
        calculate_similarity
    )
    
    assert SyntheticMetrics is not None
    assert evaluate_synthetic_quality is not None
    assert calculate_similarity is not None
