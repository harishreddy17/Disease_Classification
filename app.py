# main.py
from fastapi import FastAPI, UploadFile, File, Request
from src.predictor import Predictor
from src.logger import logger
from config import load_config
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

# -------------------------------
# Load dynamic config and model
# -------------------------------
config = load_config(model_name="resnet18")  # Change model dynamically
predictor = Predictor(model_path=f"{config.model_dir}/resnet_onion.pth", config=config)

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Onion Disease Classifier")

# Metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["method","endpoint","status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["endpoint"])
ERROR_COUNT = Counter("api_errors_total", "Total API errors")

# -------------------------------
# Middleware for metrics & logging
# -------------------------------
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        latency = time.time() - start_time
        REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
        REQUEST_LATENCY.labels(request.url.path).observe(latency)
        logger.info(f"{request.method} {request.url.path} completed_in={latency:.4f}s status={response.status_code}")
        return response
    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"API error: {e}")
        raise e

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
async def classify(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        # Use Predictor with dynamic preprocessing
        image_tensor = predictor.preprocess(file.file)
        class_name = predictor.predict(image_tensor)
        return {"predicted_class": class_name}
    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

# -------------------------------
# Metrics endpoint
# -------------------------------
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
