"""Data acquisition component for the digit recognition pipeline.

This module defines the component that downloads the MNIST dataset
and uploads it to MinIO storage.
"""

from kfp.dsl import component, Markdown
from typing import NamedTuple
from config.settings import TF_BASE_IMAGE, MINIO_BUCKET

@component(base_image=TF_BASE_IMAGE)
def get_data_batch() -> NamedTuple(
    "Outputs",
    [
        ("dataset_summary", Markdown), 
        ("train_samples", float),
        ("test_samples", float),
        ("dataset_version", str),
    ]
):
    """
    Downloads MNIST dataset using Keras and uploads arrays to MinIO storage.
    
    Returns:
        NamedTuple containing dataset summary markdown, train/test sample counts,
        and dataset version.
    """
    import json
    import os
    import numpy as np
    from tensorflow import keras
    from minio import Minio

    print("Getting data using Keras...")

    # Download data using Keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(f"Data loaded: x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    
    # --- Connect to MinIO ---
    # Verify MinIO connection variables - debugging output
    print("Setting up MinIO connection...")
    minio_endpoint = "localhost:9000"
    minio_access_key = "minio"
    # Modified secret key - this MUST match exactly what's configured on your MinIO server
    minio_secret_key = "minioadmin"
    minio_bucket = "mlpipeline"
    
    print(f"MinIO endpoint: {minio_endpoint}")
    print(f"MinIO access key: {minio_access_key}")
    print(f"MinIO bucket: {minio_bucket}")
    
    try:
        minio_client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False  # Set to True if using HTTPS
        )
        
        # Check if bucket exists and create it if it doesn't
        if not minio_client.bucket_exists(minio_bucket):
            print(f"Bucket {minio_bucket} does not exist, creating it...")
            minio_client.make_bucket(minio_bucket)
            print(f"Created bucket: {minio_bucket}")
        else:
            print(f"Bucket {minio_bucket} already exists")
    
    except Exception as e:
        print(f"ERROR setting up MinIO client: {e}")
        raise
    
    # Define paths
    local_tmp_dir = "/tmp"
    data_paths = {
        "x_train": "x_train",
        "y_train": "y_train",
        "x_test": "x_test",
        "y_test": "y_test"
    }
    
    local_paths = {k: os.path.join(local_tmp_dir, v + ".npy") for k, v in data_paths.items()}
    
    # Save arrays locally and upload to MinIO
    print("Saving data arrays locally and uploading to MinIO...")
    
    for data_key, array in [
        ("x_train", x_train),
        ("y_train", y_train),
        ("x_test", x_test),
        ("y_test", y_test)
    ]:
        local_path = local_paths[data_key]
        remote_path = data_paths[data_key]
        
        # Save locally
        print(f"Saving {data_key} to {local_path}")
        np.save(local_path, array)
        
        # Upload to MinIO
        try:
            print(f"Uploading {data_key} to {minio_bucket}/{remote_path}")
            minio_client.fput_object(minio_bucket, remote_path, local_path)
            print(f"Successfully uploaded {data_key}")
        except Exception as e:
            print(f"ERROR uploading {data_key}: {e}")
            raise
        
        # Clean up
        os.remove(local_path)
    
    # Prepare output metadata
    train_count = float(x_train.shape[0])
    test_count = float(x_test.shape[0])
    dataset_version = "1.0"
    
    # Create markdown content
    markdown_summary = f"""
# MNIST Dataset Overview

* **Source:** Keras Datasets (`keras.datasets.mnist.load_data()`)
* **Version:** {dataset_version}
* **Training Samples:** {int(train_count)}
* **Test Samples:** {int(test_count)}
* **Data Shapes:**
  * Training images: `{x_train.shape}`
  * Training labels: `{y_train.shape}`
  * Test images: `{x_test.shape}`
  * Test labels: `{y_test.shape}`
* **Storage:** MinIO bucket `{minio_bucket}`
"""
    
    # Format for KFP markdown artifact
    markdown_artifact_json = {
        'outputs': [{
            'type': 'markdown',
            'storage': 'inline',
            'source': markdown_summary
        }]
    }
    
    return (
        json.dumps(markdown_artifact_json),
        train_count,
        test_count,
        dataset_version
    )

@component(base_image=TF_BASE_IMAGE)
def get_latest_data():
    """
    Placeholder component for adding latest data to the dataset.
    
    In a real-world scenario, this could fetch new data from a database,
    API, or other data source and merge with existing data.
    """
    print("Component for adding latest data (placeholder)")
    # Actual implementation would fetch and integrate new data