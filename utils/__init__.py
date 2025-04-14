"""
Utility package for the digit recognition pipeline.
"""
from utils.minio_utils import (
    get_minio_client,
    upload_file_to_minio,
    download_file_from_minio,
    upload_directory_to_minio,
    check_create_bucket
)