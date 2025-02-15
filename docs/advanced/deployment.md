# Model Deployment

This guide covers deploying models trained with DeepBridge in various production environments.

## Overview

DeepBridge models can be deployed in several ways:
- REST API service
- Batch processing system
- Edge devices
- Cloud platforms

## Model Serialization

### Basic Model Export

```python
from deepbridge.model_distiller import ModelDistiller
import joblib

def export_model(distiller, path):
    """Export model to file"""
    joblib.dump(distiller, path)
    
def load_model(path):
    """Load exported model"""
    return joblib.load(path)
```

### ONNX Export

```python
import onnx
import onnxruntime
from skl2onnx import convert_sklearn

def export_to_onnx(model, X, path):
    """Export model to ONNX format"""
    # Convert model to ONNX
    onx = convert_sklearn(
        model,
        initial_types=[('float_input', FloatTensorType([None, X.shape[1]]))],
        target_opset=12
    )
    
    # Save model
    onnx.save_model(onx, path)
```

## REST API Deployment

### FastAPI Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class PredictionInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    probability: float

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Convert input to numpy array
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return PredictionOutput(probability=float(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Batch Processing

### Spark Integration

```python
from pyspark.sql import SparkSession
import pandas as pd

def batch_predict_spark(model, input_path, output_path):
    """Process large datasets using Spark"""
    # Initialize Spark
    spark = SparkSession.builder.appName("ModelPrediction").getOrCreate()
    
    # Read data
    df = spark.read.csv(input_path, header=True)
    
    # Define UDF for predictions
    @pandas_udf("double")
    def predict_udf(features_pd):
        return pd.Series(model.predict(features_pd))
    
    # Apply predictions
    result = df.withColumn("prediction", predict_udf("features"))
    
    # Save results
    result.write.csv(output_path)
```

### Airflow Pipeline

```python
from airflow import DAG
from airflow.operators.python import PythonOperator

def create_prediction_pipeline():
    """Create Airflow DAG for batch predictions"""
    dag = DAG(
        'model_predictions',
        schedule_interval='@daily'
    )
    
    def load_data():
        # Load data logic
        pass
    
    def make_predictions():
        # Prediction logic
        pass
    
    def save_results():
        # Save results logic
        pass
    
    with dag:
        load_task = PythonOperator(
            task_id='load_data',
            python_callable=load_data
        )
        
        predict_task = PythonOperator(
            task_id='make_predictions',
            python_callable=make_predictions
        )
        
        save_task = PythonOperator(
            task_id='save_results',
            python_callable=save_results
        )
        
        load_task >> predict_task >> save_task
        
    return dag
```

## Cloud Deployment

### AWS Lambda

```python
import json
import boto3
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler for model predictions"""
    try:
        # Parse input
        features = json.loads(event['body'])
        
        # Load model from S3 if needed
        if not hasattr(lambda_handler, 'model'):
            s3 = boto3.client('s3')
            s3.download_file('model-bucket', 'model.joblib', '/tmp/model.joblib')
            lambda_handler.model = joblib.load('/tmp/model.joblib')
        
        # Make prediction
        prediction = lambda_handler.model.predict([features])[0]
        
        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': float(prediction)})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Google Cloud Functions

```python
from google.cloud import storage
import functions_framework

@functions_framework.http
def predict(request):
    """Google Cloud Function for predictions"""
    try:
        # Get input data
        request_json = request.get_json(silent=True)
        features = request_json['features']
        
        # Load model if needed
        if not hasattr(predict, 'model'):
            storage_client = storage.Client()
            bucket = storage_client.bucket('model-bucket')
            blob = bucket.blob('model.joblib')
            blob.download_to_filename('/tmp/model.joblib')
            predict.model = joblib.load('/tmp/model.joblib')
        
        # Make prediction
        prediction = predict.model.predict([features])[0]
        
        return {'prediction': float(prediction)}
    except Exception as e:
        return {'error': str(e)}, 500
```

## Model Monitoring

### Prediction Monitoring

```python
import logging
from datetime import datetime

class MonitoredModel:
    """Wrapper for model monitoring"""
    
    def __init__(self, model):
        self.model = model
        self.logger = logging.getLogger(__name__)
        
    def predict(self, features):
        """Make predictions with monitoring"""
        start_time = datetime.now()
        
        try:
            prediction = self.model.predict(features)
            
            # Log prediction
            self.logger.info({
                'timestamp': datetime.now(),
                'prediction': float(prediction),
                'features': features.tolist(),
                'latency': (datetime.now() - start_time).total_seconds()
            })
            
            return prediction
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
```

### Performance Tracking

```python
from prometheus_client import Counter, Histogram

class MetricsCollector:
    """Collect model performance metrics"""
    
    def __init__(self):
        self.prediction_count = Counter(
            'model_predictions_total',
            'Total number of predictions'
        )
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Time spent making predictions'
        )
        self.prediction_errors = Counter(
            'model_prediction_errors_total',
            'Total number of prediction errors'
        )
```

## Best Practices

1. **Model Versioning**
   ```python
   def deploy_model_version(model, version):
       """Deploy new model version"""
       # Save model with version
       path = f"models/v{version}/model.joblib"
       joblib.dump(model, path)
       
       # Update version metadata
       metadata = {
           'version': version,
           'deployed_at': datetime.now().isoformat(),
           'features': model.feature_names,
           'metrics': model.get_metrics()
       }
       
       return metadata
   ```

2. **Health Checks**
   ```python
   def model_health_check():
       """Check model health"""
       try:
           # Make test prediction
           test_features = np.random.rand(1, n_features)
           prediction = model.predict(test_features)
           
           # Check prediction shape and values
           assert prediction.shape == (1,)
           assert 0 <= prediction <= 1
           
           return True
       except Exception as e:
           logging.error(f"Health check failed: {str(e)}")
           return False
   ```

3. **Error Handling**
   ```python
   def safe_predict(model, features):
       """Make predictions with error handling"""
       try:
           # Validate input
           if not isinstance(features, np.ndarray):
               features = np.array(features)
           
           if features.shape[1] != model.n_features:
               raise ValueError("Invalid feature dimension")
           
           # Make prediction
           return model.predict(features)
       except Exception as e:
           logging.error(f"Prediction error: {str(e)}")
           raise
   ```

## Next Steps

- Review [Model Validation](validation.md) for model preparation
- Check [Optimization](optimization.md) for performance tuning
- See [API Reference](../api/model_validation.md) for detailed documentation