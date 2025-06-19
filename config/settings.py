# config/settings.py
"""Configuration settings for the Kubeflow pipeline.

This module centralizes all configuration settings for the pipeline,
making it easier to update endpoints, credentials, and other parameters.
"""
import os

# MinIO configuration
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "mlpipeline")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"

# MLflow configuration
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", 
    "http://mlflow-server.kubeflow.svc.cluster.local:5000"
)
MLFLOW_S3_ENDPOINT_URL = os.environ.get(
    "MLFLOW_S3_ENDPOINT_URL", 
    f"http://{MINIO_ENDPOINT}"
)
MLFLOW_BUCKET = os.environ.get("MLFLOW_BUCKET", "mlflow")

# Kubernetes/KServe configuration
K8S_NAMESPACE = os.environ.get("K8S_NAMESPACE", "kubeflow")
KSERVE_SA_NAME = os.environ.get("KSERVE_SA_NAME", "sa-minio-kserve")

# Model configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "digits-recognizer-model")
MODEL_SAVE_PREFIX = os.environ.get("MODEL_SAVE_PREFIX", "models/detect-digits")

# Pipeline settings
PIPELINE_NAME = os.environ.get("PIPELINE_NAME", "digits-recognizer-mlflow-pipeline")
DEFAULT_EXPERIMENT_NAME = os.environ.get("DEFAULT_EXPERIMENT_NAME", "digits-pipeline")

# Base image for components
TF_BASE_IMAGE = os.environ.get(
    "TF_BASE_IMAGE", 
    "public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-tensorflow-full:v1.5.0"
)