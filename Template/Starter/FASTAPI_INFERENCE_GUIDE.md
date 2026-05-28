# 🚀 FastAPI Inference API Guide

Complete production-ready REST API for the breast cancer detector using FastAPI.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn
# Or update everything:
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
cd d:\Mlops\Template\Starter
python inference_api.py
```

You'll see:
```
🚀 Starting Breast Cancer Detector API...
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. View Interactive Docs
Open your browser: **http://localhost:8000/docs**

You'll see the **Swagger UI** with all endpoints documented and testable.

---

## API Endpoints

### **GET /health**
Check if API and model are ready.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "sklearn_classifier",
  "model_version": "production"
}
```

---

### **POST /predict**
Make a single patient prediction.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.905,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave_points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.002812,
    "radius_worst": 25.38,
    "texture_worst": 17.6,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.162,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189
  }'
```

**Response:**
```json
{
  "success": true,
  "diagnosis": "cancer",
  "diagnosis_code": 1,
  "confidence": 0.956,
  "probability_no_cancer": 0.044,
  "probability_cancer": 0.956,
  "model_version": "production",
  "message": "Patient has cancer with 95.60% confidence"
}
```

**What the fields mean:**
- `diagnosis` - "cancer" or "no_cancer"
- `confidence` - How sure is the model? (0-1 scale)
- `probability_cancer` - Model thinks it's cancer with this probability
- `probability_no_cancer` - Model thinks it's no cancer with this probability

---

### **POST /predict-batch**
Make predictions for multiple patients at once.

**Request:**
```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "radius_mean": 17.99,
      "texture_mean": 10.38,
      ...all 30 measurements...
    },
    {
      "radius_mean": 16.0,
      "texture_mean": 12.5,
      ...all 30 measurements...
    }
  ]'
```

**Response:**
```json
{
  "success": true,
  "batch_size": 2,
  "predictions": [
    {
      "patient_index": 0,
      "diagnosis": "cancer",
      "confidence": 0.956,
      "probability_cancer": 0.956
    },
    {
      "patient_index": 1,
      "diagnosis": "no_cancer",
      "confidence": 0.842,
      "probability_cancer": 0.158
    }
  ]
}
```

---

### **GET /model-info**
Get information about the loaded model.

**Request:**
```bash
curl http://localhost:8000/model-info
```

**Response:**
```json
{
  "model_name": "sklearn_classifier",
  "model_version": "production",
  "model_type": "RandomForestClassifier",
  "preprocessing_metadata": {
    "random_state": 17,
    "target": "target"
  },
  "expected_features": 30
}
```

---

## Python Client Example

```python
import requests
import json

API_URL = "http://localhost:8000"

# Patient data
patient = {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.905,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave_points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.002812,
    "radius_worst": 25.38,
    "texture_worst": 17.6,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.162,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189
}

# Make prediction
response = requests.post(f"{API_URL}/predict", json=patient)
result = response.json()

print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Message: {result['message']}")
```

---

## Docker Deployment

### Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "inference_api.py"]
```

### Build and Run
```bash
# Build
docker build -t cancer-detector-api:latest .

# Run
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  cancer-detector-api:latest
```

---

## Production Features

### ✅ What's Included

1. **Automatic Model Loading** - Loads production model on startup
2. **Error Handling** - Graceful error messages and HTTP status codes
3. **Logging** - All predictions logged for monitoring
4. **MLflow Integration** - Each prediction logged to MLflow
5. **Batch Predictions** - Handle multiple patients efficiently
6. **Health Checks** - Monitor API and model status
7. **Interactive Docs** - Swagger UI at /docs
8. **Type Validation** - Pydantic validates all inputs
9. **CORS Ready** - Can add CORS middleware easily
10. **Async Ready** - FastAPI handles concurrent requests

---

## Monitoring

### Check MLflow Logs
All predictions are logged to MLflow at `http://localhost:5000`

Go to **Inference** experiment to see:
- API predictions made
- Batch predictions
- Confidence levels
- Diagnosis distribution

### Example Queries

**View all API predictions:**
```
experiment: inference
run_name: api_prediction
```

**View batch predictions:**
```
experiment: inference  
run_name: api_batch_prediction
metrics: batch_size > 1
```

---

## Load Testing

Test API with Apache Bench:

```bash
# Single request
ab -n 100 -c 10 http://localhost:8000/health

# Batch predictions
ab -n 50 -c 5 -p patient_data.json http://localhost:8000/predict
```

---

## Error Handling

### Model Not Loaded
**Status:** 503 Service Unavailable
```json
{
  "detail": "Model not loaded. Please try again later."
}
```

### Invalid Patient Data
**Status:** 422 Unprocessable Entity
```json
{
  "detail": [
    {
      "loc": ["body", "radius_mean"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Prediction Error
**Status:** 500 Internal Server Error
```json
{
  "detail": "Prediction failed: [error details]"
}
```

---

## Configuration

### Change API Port
```bash
# In inference_api.py, change the port in main()
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,  # Change here
        log_level="info"
    )
```

### Change MLflow URI
```bash
# Set environment variable before running
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
python inference_api.py
```

---

## Integration Examples

### Django Integration
```python
# In your Django view
import requests

def predict_cancer(request):
    patient_data = request.POST.get('data')
    response = requests.post('http://localhost:8000/predict', json=patient_data)
    return JsonResponse(response.json())
```

### NodeJS Integration
```javascript
// In your Express server
app.post('/api/predict', async (req, res) => {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req.body)
  });
  const result = await response.json();
  res.json(result);
});
```

### Database Integration
```python
# Save predictions to database
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('postgresql://user:pass@localhost/predictions_db')
Session = sessionmaker(bind=engine)
session = Session()

# After making prediction
prediction = Prediction(
    patient_id=1,
    diagnosis=result['diagnosis'],
    confidence=result['confidence'],
    timestamp=datetime.now()
)
session.add(prediction)
session.commit()
```

---

## Performance Tips

1. **Batch Predictions** - Use `/predict-batch` for multiple patients (faster than individual calls)
2. **Model Caching** - Model is loaded once at startup and reused
3. **Async Processing** - FastAPI handles concurrent requests efficiently
4. **Database Indexing** - If storing predictions, index on timestamp and patient_id

---

## Security Best Practices

1. **Add Authentication** - Use API keys or JWT tokens
2. **Rate Limiting** - Limit requests per IP/user
3. **HTTPS** - Use SSL certificates in production
4. **Input Validation** - Pydantic validates all inputs automatically
5. **Logging** - Log all requests for audit trail
6. **CORS** - Restrict which domains can call the API

Example with authentication:
```python
from fastapi.security import APIKey, HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(patient: PatientData, credentials: APIKey = Depends(security)):
    # Verify API key
    if credentials.credentials != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of code
```

---

## Summary

✅ **Production-Ready** - Fully functional inference API  
✅ **Monitored** - All predictions logged to MLflow  
✅ **Scalable** - Handles batch and concurrent requests  
✅ **Documented** - Interactive Swagger UI  
✅ **Tested** - Health checks and error handling  
✅ **Extensible** - Easy to add authentication, logging, databases

**Start the API:**
```bash
python inference_api.py
```

**Test it:**
```bash
curl http://localhost:8000/health
```

**View docs:**
```
http://localhost:8000/docs
```

🚀 Your breast cancer detector is now production-ready!

---

*Need help with deployment? Check the Docker section or ask for specific platform guidance (AWS, Azure, GCP, etc.)*
