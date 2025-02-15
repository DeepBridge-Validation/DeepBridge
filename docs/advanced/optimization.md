# Performance Optimization

This guide covers optimization techniques for both model validation and distillation in DeepBridge, focusing on performance, efficiency, and resource utilization.

## Model Optimization

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from deepbridge.model_distiller import ModelDistiller

def optimize_distiller_params(X, probas, model_type="gbm"):
    """Optimize distiller hyperparameters"""
    
    # Define parameter grid based on model type
    param_grids = {
        "gbm": {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 4, 5]
        },
        "xgb": {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
    }
    
    # Create and train distiller with grid search
    distiller = ModelDistiller(model_type=model_type)
    grid_search = GridSearchCV(
        distiller.model,
        param_grids[model_type],
        cv=5,
        scoring='neg_mean_squared_error'
    )
    
    grid_search.fit(X, probas)
    return grid_search.best_params_
```

### Model Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

def calibrate_distilled_model(distiller, X, y):
    """Calibrate model probabilities"""
    calibrated_model = CalibratedClassifierCV(
        distiller.model,
        cv=5,
        method='isotonic'
    )
    
    calibrated_model.fit(X, y)
    return calibrated_model
```

## Performance Optimization

### Memory Management

```python
class MemoryOptimizedDistiller(ModelDistiller):
    """Memory-efficient model distiller"""
    
    def fit(self, X, probas, chunk_size=1000):
        """Fit model using chunked data processing"""
        from sklearn.utils import gen_batches
        
        # Process data in chunks
        for batch_idx in gen_batches(X.shape[0], chunk_size):
            X_batch = X[batch_idx]
            probas_batch = probas[batch_idx]
            
            # Train on batch
            self.model.partial_fit(X_batch, probas_batch)
        
        return self

def optimize_memory_usage(X, dtype_map=None):
    """Optimize memory usage of input data"""
    if dtype_map is None:
        dtype_map = {
            'float64': 'float32',
            'int64': 'int32'
        }
    
    for col in X.columns:
        if X[col].dtype.name in dtype_map:
            X[col] = X[col].astype(dtype_map[X[col].dtype.name])
    
    return X
```

### Computational Efficiency

```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ParallelDistiller(ModelDistiller):
    """Parallel processing implementation"""
    
    def predict(self, X, n_jobs=-1):
        """Make predictions using parallel processing"""
        def predict_chunk(chunk):
            return self.model.predict_proba(chunk)
        
        # Split data into chunks
        chunks = np.array_split(X, max(1, min(len(X) // 1000, n_jobs)))
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            predictions = list(executor.map(predict_chunk, chunks))
        
        return np.vstack(predictions)
```

## Resource Optimization

### GPU Utilization

```python
try:
    import cupy as cp
    import cudf
    
    class GPUDistiller(ModelDistiller):
        """GPU-accelerated model distiller"""
        
        def fit(self, X, probas):
            """Fit model using GPU acceleration"""
            # Convert data to GPU
            X_gpu = cudf.DataFrame(X)
            probas_gpu = cp.array(probas)
            
            # Train model
            self.model.fit(X_gpu, probas_gpu)
            return self
            
except ImportError:
    print("GPU acceleration requires cupy and cudf")
```

### Multi-Processing

```python
from multiprocessing import Pool

def parallel_model_training(models, X, y, n_processes=4):
    """Train multiple models in parallel"""
    def train_model(model_data):
        model, X, y = model_data
        model.fit(X, y)
        return model
    
    # Prepare training data
    training_data = [(model, X, y) for model in models]
    
    # Train models in parallel
    with Pool(n_processes) as pool:
        trained_models = pool.map(train_model, training_data)
    
    return trained_models
```

## Monitoring and Profiling

### Performance Monitoring

```python
import time
import psutil
import logging

class MonitoredDistiller(ModelDistiller):
    """Distiller with performance monitoring"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_metrics = {
            'training_time': [],
            'memory_usage': [],
            'prediction_time': []
        }
    
    def fit(self, X, probas):
        """Fit model with performance monitoring"""
        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss
        
        # Train model
        super().fit(X, probas)
        
        # Record metrics
        self.performance_metrics['training_time'].append(
            time.time() - start_time
        )
        self.performance_metrics['memory_usage'].append(
            psutil.Process().memory_info().rss - memory_start
        )
        
        return self
```

### Resource Profiling

```python
import cProfile
import pstats

def profile_model_performance(model, X, y):
    """Profile model performance"""
    profiler = cProfile.Profile()
    
    # Profile training
    profiler.enable()
    model.fit(X, y)
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
    
    return stats
```

## Best Practices

### 1. Data Optimization

```python
def optimize_dataset(X, y):
    """Optimize dataset for performance"""
    # Convert to efficient types
    X = optimize_memory_usage(X)
    
    # Remove redundant features
    correlation_threshold = 0.95
    correlations = X.corr().abs()
    to_drop = set()
    
    for i in range(len(correlations.columns)):
        for j in range(i):
            if correlations.iloc[i, j] > correlation_threshold:
                to_drop.add(correlations.columns[i])
    
    X = X.drop(columns=list(to_drop))
    
    return X, y
```

### 2. Model Configuration

```python
def configure_model_for_performance(model_type, data_size):
    """Configure model based on data size"""
    if data_size < 10000:
        return {
            'gbm': {'n_estimators': 50},
            'xgb': {'n_estimators': 100},
            'mlp': {'hidden_layer_sizes': (50, 25)}
        }[model_type]
    else:
        return {
            'gbm': {'n_estimators': 100},
            'xgb': {'n_estimators': 200},
            'mlp': {'hidden_layer_sizes': (100, 50)}
        }[model_type]
```

### 3. Resource Management

```python
def manage_resources(data_size, available_memory):
    """Calculate optimal chunk size and workers"""
    chunk_size = min(1000, data_size // 10)
    n_workers = min(
        psutil.cpu_count(),
        max(1, available_memory // (chunk_size * 8))
    )
    
    return chunk_size, n_workers
```

## Optimization Tips

1. **Memory Efficiency**
   - Use appropriate data types
   - Process large datasets in chunks
   - Clean up unused objects

2. **Computational Performance**
   - Use parallel processing when appropriate
   - Optimize feature selection
   - Cache intermediate results

3. **Resource Utilization**
   - Monitor memory usage
   - Profile critical operations
   - Use GPU acceleration when available

## Next Steps

- Check [Model Validation](validation.md) for experiment management
- See [Custom Models](custom_models.md) for model implementation
- Review [API Reference](../api/model_validation.md) for detailed documentation