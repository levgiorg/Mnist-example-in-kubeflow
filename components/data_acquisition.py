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
    
    # Connect to MinIO
    minio_client = Minio(
        "localhost:9000",  # In production, use env vars
        access_key="minio",
        secret_key="your-secret-key",
        secure=False
    )
    minio_bucket = "mlpipeline"
    
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
    print("Uploading data to MinIO...")
    
    for data_key, array in [
        ("x_train", x_train),
        ("y_train", y_train),
        ("x_test", x_test),
        ("y_test", y_test)
    ]:
        local_path = local_paths[data_key]
        remote_path = data_paths[data_key]
        
        # Save locally
        np.save(local_path, array)
        
        # Upload to MinIO
        minio_client.fput_object(minio_bucket, remote_path, local_path)
        print(f"Uploaded {data_key} to {minio_bucket}/{remote_path}")
        
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