"""Utility functions for MinIO operations.

This module provides helper functions for common MinIO operations
used across different pipeline components.
"""

import os
from minio import Minio
from config.settings import (
    MINIO_ENDPOINT, 
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    MINIO_BUCKET
)

def get_minio_client():
    """Create and return a configured MinIO client.
    
    Returns:
        Minio: Configured MinIO client ready for use
    """
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

def upload_file_to_minio(local_path, object_name, bucket_name=MINIO_BUCKET):
    """Upload a local file to MinIO.
    
    Args:
        local_path: Path to local file
        object_name: Name to use in MinIO bucket
        bucket_name: Target bucket name (defaults to config setting)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        client = get_minio_client()
        client.fput_object(bucket_name, object_name, local_path)
        return True
    except Exception as e:
        print(f"Error uploading {local_path} to {bucket_name}/{object_name}: {e}")
        return False

def download_file_from_minio(object_name, local_path, bucket_name=MINIO_BUCKET):
    """Download a file from MinIO to local storage.
    
    Args:
        object_name: Name of object in MinIO bucket
        local_path: Path where to save locally
        bucket_name: Source bucket name (defaults to config setting)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        client = get_minio_client()
        client.fget_object(bucket_name, object_name, local_path)
        return True
    except Exception as e:
        print(f"Error downloading {bucket_name}/{object_name} to {local_path}: {e}")
        return False

def upload_directory_to_minio(local_dir, prefix, bucket_name=MINIO_BUCKET):
    """Upload a directory and its contents to MinIO recursively.
    
    Args:
        local_dir: Local directory to upload
        prefix: Prefix path in MinIO bucket
        bucket_name: Target bucket name (defaults to config setting)
        
    Returns:
        tuple: (success_count, error_count)
    """
    success_count = 0
    error_count = 0
    client = get_minio_client()
    
    for root, _, files in os.walk(local_dir):
        for file in files:
            # Get full path
            local_file_path = os.path.join(root, file)
            
            # Get relative path from the source directory
            rel_path = os.path.relpath(local_file_path, local_dir)
            
            # Construct MinIO object path with normalized separators
            minio_key = f"{prefix}/{rel_path}".replace("\\", "/")
            
            try:
                client.fput_object(bucket_name, minio_key, local_file_path)
                success_count += 1
            except Exception as e:
                print(f"Error uploading {local_file_path}: {e}")
                error_count += 1
    
    return success_count, error_count

def check_create_bucket(bucket_name=MINIO_BUCKET):
    """Check if a bucket exists and create it if it doesn't.
    
    Args:
        bucket_name: Name of bucket to check/create
        
    Returns:
        bool: True if bucket exists or was created, False on error
    """
    try:
        client = get_minio_client()
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Created bucket: {bucket_name}")
        return True
    except Exception as e:
        print(f"Error checking/creating bucket {bucket_name}: {e}")
        return False