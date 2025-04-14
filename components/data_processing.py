"""Data processing component for the digit recognition pipeline.

This module defines the component that reshapes and normalizes
the MNIST data for training.
"""

from kfp.dsl import component, Markdown
from typing import NamedTuple
from config.settings import TF_BASE_IMAGE

@component(base_image=TF_BASE_IMAGE)
def reshape_data() -> NamedTuple("Outputs", [("reshape_summary", Markdown)]): 
    """
    Reshape and normalize MNIST data for model training.
    
    - Reshapes data to add channel dimension (28x28x1)
    - Normalizes pixel values to [0,1] range
    - Saves processed data back to MinIO
    
    Returns:
        NamedTuple containing summary of the preprocessing steps.
    """
    import json
    import os
    import numpy as np
    from minio import Minio

    print("Reshaping and normalizing MNIST data...")

    # MinIO configuration - make sure these match your actual values
    # IMPORTANT: These must match the values used in get_data_batch
    MINIO_ENDPOINT = "localhost:9000"
    MINIO_ACCESS_KEY = "minio"
    MINIO_SECRET_KEY = "minioadmin"
    MINIO_BUCKET = "mlpipeline"
    MINIO_SECURE = False

    # Debug output
    print("Setting up MinIO connection...")
    print(f"MinIO endpoint: {MINIO_ENDPOINT}")
    print(f"MinIO access key: {MINIO_ACCESS_KEY}")
    print(f"MinIO bucket: {MINIO_BUCKET}")
    print(f"MinIO secure: {MINIO_SECURE}")

    local_tmp_dir = "/tmp"

    # Define paths
    minio_paths = {
        "x_train": "x_train",
        "x_test": "x_test"
    }
    
    local_paths = {k: os.path.join(local_tmp_dir, v + ".npy") for k, v in minio_paths.items()}
    
    # Create MinIO client
    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        
        # Check if bucket exists
        if not minio_client.bucket_exists(MINIO_BUCKET):
            print(f"Error: Bucket {MINIO_BUCKET} does not exist")
            raise ValueError(f"Bucket {MINIO_BUCKET} not found")
        else:
            print(f"Successfully connected to bucket: {MINIO_BUCKET}")
    except Exception as e:
        print(f"Error connecting to MinIO: {e}")
        raise
    
    # Load and process data
    x_train_shape, x_test_shape = None, None
    
    try:
        # Process training data
        print(f"Loading training data from {MINIO_BUCKET}/{minio_paths['x_train']}")
        minio_client.fget_object(MINIO_BUCKET, minio_paths["x_train"], local_paths["x_train"])
        x_train = np.load(local_paths["x_train"])
        print(f"Original x_train shape: {x_train.shape}")
        x_train_shape = x_train.shape
        
        # Reshape and normalize
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_train = x_train / 255.0
        print(f"Processed x_train shape: {x_train.shape}")
        
        # Save back to MinIO
        np.save(local_paths["x_train"], x_train)
        print(f"Uploading processed training data back to MinIO")
        minio_client.fput_object(MINIO_BUCKET, minio_paths["x_train"], local_paths["x_train"])
        print("Training data successfully processed and uploaded")
        
        # Process test data
        print(f"Loading test data from {MINIO_BUCKET}/{minio_paths['x_test']}")
        minio_client.fget_object(MINIO_BUCKET, minio_paths["x_test"], local_paths["x_test"])
        x_test = np.load(local_paths["x_test"])
        print(f"Original x_test shape: {x_test.shape}")
        x_test_shape = x_test.shape
        
        # Reshape and normalize
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_test = x_test / 255.0
        print(f"Processed x_test shape: {x_test.shape}")
        
        # Save back to MinIO
        np.save(local_paths["x_test"], x_test)
        print(f"Uploading processed test data back to MinIO")
        minio_client.fput_object(MINIO_BUCKET, minio_paths["x_test"], local_paths["x_test"])
        print("Test data successfully processed and uploaded")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        raise
    finally:
        # Clean up local files
        for path in local_paths.values():
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed temporary file: {path}")
    
    # Create markdown summary
    markdown_summary = f"""
# Data Preprocessing Summary

## Preprocessing Steps:
1. **Reshaping:** Added channel dimension for CNN input
   * Training data: {x_train_shape} → {(-1, 28, 28, 1)}
   * Test data: {x_test_shape} → {(-1, 28, 28, 1)}
2. **Normalization:** Scaled pixel values from [0,255] to [0,1]
   * Divided all pixel values by 255.0

## Storage:
* Processed data saved back to MinIO bucket: `{MINIO_BUCKET}`
* Original paths were preserved: `{", ".join(minio_paths.values())}`
"""
    
    markdown_artifact_json = {
        'outputs': [{
            'type': 'markdown',
            'storage': 'inline',
            'source': markdown_summary
        }]
    }
    
    return (json.dumps(markdown_artifact_json),)