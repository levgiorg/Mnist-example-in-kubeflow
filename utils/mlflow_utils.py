# utils/mlflow_utils.py
"""MLflow utility functions for experiment tracking and model management.

This module provides helper functions for MLflow operations used
across different pipeline components.
"""

import os
import boto3
import mlflow
from mlflow.tracking import MlflowClient
from config.settings import (
    MLFLOW_TRACKING_URI,
    MLFLOW_S3_ENDPOINT_URL,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MLFLOW_BUCKET
)

def setup_mlflow_environment():
    """Set up MLflow environment variables and configuration.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Set environment variables
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
        os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
        os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
        
        # Configure MLflow client
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        return True
    except Exception as e:
        print(f"Error setting up MLflow environment: {e}")
        return False

def ensure_mlflow_bucket_exists():
    """Ensure the MLflow artifacts bucket exists in S3/MinIO.
    
    Returns:
        bool: True if bucket exists or was created, False on error
    """
    try:
        # Create S3 client for bucket operations
        s3_client = boto3.client(
            "s3",
            endpoint_url=MLFLOW_S3_ENDPOINT_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=boto3.session.Config(signature_version="s3v4"),
        )
        
        # Check if bucket exists
        buckets_response = s3_client.list_buckets()
        existing_buckets = [bucket["Name"] for bucket in buckets_response["Buckets"]]
        
        if MLFLOW_BUCKET not in existing_buckets:
            print(f"Creating MLflow bucket: {MLFLOW_BUCKET}")
            s3_client.create_bucket(Bucket=MLFLOW_BUCKET)
            print(f"MLflow bucket '{MLFLOW_BUCKET}' created successfully")
        else:
            print(f"MLflow bucket '{MLFLOW_BUCKET}' already exists")
            
        return True
    except Exception as e:
        print(f"Error checking/creating MLflow bucket: {e}")
        return False

def get_or_create_experiment(experiment_name):
    """Get or create an MLflow experiment.
    
    Args:
        experiment_name: Name of the experiment to get or create
        
    Returns:
        str: Experiment ID if successful, None otherwise
    """
    try:
        # Ensure MLflow is properly configured
        setup_mlflow_environment()
        
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
            
        return experiment_id
    except Exception as e:
        print(f"Error getting/creating MLflow experiment: {e}")
        return None

def list_registered_models():
    """List all models in the MLflow Model Registry.
    
    Returns:
        list: List of registered model names
    """
    try:
        client = MlflowClient()
        models = client.list_registered_models()
        return [model.name for model in models]
    except Exception as e:
        print(f"Error listing registered models: {e}")
        return []

def get_latest_model_version(model_name):
    """Get the latest version of a registered model.
    
    Args:
        model_name: Name of the registered model
        
    Returns:
        dict: Model version details if successful, None otherwise
    """
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["None"])
        if versions:
            version = versions[0]
            return {
                "version": version.version,
                "run_id": version.run_id,
                "status": version.status,
                "stage": version.current_stage
            }
        return None
    except Exception as e:
        print(f"Error getting latest model version: {e}")
        return None

def transition_model_to_stage(model_name, version, stage):
    """Transition a model version to a specified stage.
    
    Args:
        model_name: Name of the registered model
        version: Version number to transition
        stage: Target stage (e.g., 'Staging', 'Production')
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"Model {model_name} version {version} transitioned to {stage}")
        return True
    except Exception as e:
        print(f"Error transitioning model to stage: {e}")
        return False

def log_model_metadata(run_id, metadata_dict):
    """Log additional metadata to an MLflow run.
    
    Args:
        run_id: MLflow run ID
        metadata_dict: Dictionary of metadata key-value pairs
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        client = MlflowClient()
        for key, value in metadata_dict.items():
            client.set_tag(run_id, key, value)
        return True
    except Exception as e:
        print(f"Error logging metadata: {e}")
        return False